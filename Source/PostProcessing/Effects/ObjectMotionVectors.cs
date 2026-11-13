using System;
using System.Collections.Generic;
using TUFX;

namespace UnityEngine.Rendering.PostProcessing
{
    /// <summary>
    /// Screen-space motion over a camera exposure for the KSP flight camera, written in the format of
    /// Unity's motion vector texture (viewport-UV displacement) so the stock motion blur passes consume it.
    ///
    /// Unity's own motion vectors are a per-frame difference of transforms. KSP moves vessels only in
    /// physics steps (50 Hz, no rigidbody interpolation) and, with Krakensbane engaged, moves the world
    /// instead of the active vessel, so above 50 fps most rendered frames carry zero object motion and
    /// only camera motion blurs. Here the motion comes from velocities instead:
    /// - background (terrain, ocean, buildings, sky): camera motion relative to the main body's frame,
    ///   reprojected from depth with the built-in camera motion vector pass;
    /// - loaded vessels: each part's rigidbody velocity and angular velocity over the exposure, drawn
    ///   with the built-in object motion vector pass and a past model matrix.
    /// Both use Hidden/Internal-MotionVectors, whose matrices are plain shader globals.
    /// </summary>
    internal sealed class ObjectMotionVectors
    {
        const float QuadDepth = 1f;           // background quad plane, camera space; clip w = 1 keeps its uv exact
        const float CutSpeed = 4000f;         // m/s: faster camera moves are cuts, not motion
        const float CutAngularSpeed = 1440f;  // deg/s
        const float CacheRefresh = 2f;        // seconds between renderer list refreshes per vessel
        const float MinPixels = 1.5f;         // renderers with a smaller screen radius are not drawn

        static readonly int PreviousVP = Shader.PropertyToID("_PreviousVP");
        static readonly int NonJitteredVP = Shader.PropertyToID("_NonJitteredVP");
        static readonly int PreviousM = Shader.PropertyToID("_PreviousM");
        static readonly int HasLastPositionData = Shader.PropertyToID("_HasLastPositionData");
        static readonly int ForceNoMotion = Shader.PropertyToID("_ForceNoMotion");
        static readonly int MotionVectorDepthBias = Shader.PropertyToID("_MotionVectorDepthBias");

        static Material s_Material;
        static bool s_MaterialMissing;

        static Material Material
        {
            get
            {
                if (s_Material == null && !s_MaterialMissing)
                {
                    var shader = Shader.Find("Hidden/Internal-MotionVectors");
                    if (shader == null || !shader.isSupported)
                    {
                        s_MaterialMissing = true;
                        Debug.LogWarning("[TUFX] Hidden/Internal-MotionVectors unavailable; motion blur uses Unity motion vectors");
                    }
                    else
                        s_Material = new Material(shader) { hideFlags = HideFlags.DontSave };
                }
                return s_Material;
            }
        }

        struct Sample
        {
            public double time;        // Time.fixedTime
            public Vector3d position;  // relative to the main body
            public Quaternion rotation;
        }

        sealed class VesselEntry
        {
            public int partCount = -1;
            public float builtAt = float.NegativeInfinity;
            public Renderer[] renderers = new Renderer[0];
            public int[][] submeshes = new int[0][];   // opaque submesh indices per renderer
            public float[] radius = new float[0];      // world-space bounds radius per renderer
        }

        Sample m_Sample;             // camera at the last physics step
        bool m_HasSample;
        int m_LastFrame = -100;
        Vector3 m_VelocityBody;      // camera velocity in the main body frame, world axes
        Vector3 m_AngularVelocity;   // rad/s, world axes
        Mesh m_Quad;
        Vector2 m_QuadTan;
        readonly Vector3[] m_QuadVerts = new Vector3[4];
        readonly Plane[] m_Planes = new Plane[6];
        readonly Dictionary<Vessel, VesselEntry> m_Vessels = new Dictionary<Vessel, VesselEntry>();
        readonly List<Vessel> m_Stale = new List<Vessel>();
        readonly List<Material> m_MatBuf = new List<Material>();
        readonly List<int> m_SubBuf = new List<int>();
        readonly System.Diagnostics.Stopwatch m_Watch = new System.Diagnostics.Stopwatch();

        // Debug readback (TUFX_CONFIGURATION MotionVectorCheck)
        RenderTexture m_CheckRT, m_ColorRT;
        float m_NextCheck, m_NextColorCheck;
        bool m_CheckPending, m_ColorCheckPending;
        readonly List<CheckPoint> m_CheckPoints = new List<CheckPoint>();
        double m_BuildMs; int m_BuildFrames, m_Draws, m_SkinnedDraws;

        struct CheckPoint
        {
            public string label;
            public Vector2 pixel;
            public Vector2 expected;   // pixels over the exposure
        }

        /// <summary>True when velocities can be generated for this camera.</summary>
        public static bool Supported(Camera camera)
        {
            if (!HighLogic.LoadedSceneIsFlight || FlightCamera.fetch == null || camera != FlightCamera.fetch.mainCamera)
                return false;
            if (camera.nearClipPlane >= QuadDepth)
                return false;
            if (camera.actualRenderingPath == RenderingPath.Forward && camera.allowMSAA && QualitySettings.antiAliasing > 1)
                return false;   // MSAA camera depth cannot be bound with a single-sample target
            return Material != null;
        }

        public void Release()
        {
            if (m_Quad != null) Object.Destroy(m_Quad);
            m_Quad = null;
            if (m_CheckRT != null) m_CheckRT.Release();
            m_CheckRT = null;
            if (m_ColorRT != null) m_ColorRT.Release();
            m_ColorRT = null;
            m_Vessels.Clear();
        }

        /// <summary>
        /// Records exposure-long screen motion into <paramref name="target"/> (RGHalf, screen size, viewport-UV units).
        /// <paramref name="cmd"/> must execute inside the camera's render with its matrices and depth buffer current
        /// (PostProcessLayer's AfterForwardAlpha buffer).
        /// </summary>
        public void Render(PostProcessRenderContext context, CommandBuffer cmd, int target, float exposure)
        {
            var camera = context.camera;
            m_Watch.Reset();
            m_Watch.Start();

            UpdateCameraMotion(camera);

            cmd.BeginSample("TUFX Motion Vectors");
            cmd.SetRenderTarget(target, BuiltinRenderTextureType.CameraTarget);
            cmd.ClearRenderTarget(false, true, Color.clear);

            var transform = camera.transform;
            Vector3 position = transform.position;
            Quaternion rotation = transform.rotation;
            Matrix4x4 projection = camera.nonJitteredProjectionMatrix;
            Matrix4x4 viewProjection = projection * camera.worldToCameraMatrix;
            Vector3 frameVelocity = Krakensbane.GetFrameVelocityV3f();
            Vector3 cameraVelocity = m_VelocityBody - frameVelocity;  // Unity frame: the world moves at -frameVelocity
            Quaternion pastRotation = PastRotation(rotation, m_AngularVelocity, exposure);
            Matrix4x4 pastObjectVP = projection * WorldToCamera(position - cameraVelocity * exposure, pastRotation);
            Matrix4x4 pastBackgroundVP = projection * WorldToCamera(position - m_VelocityBody * exposure, pastRotation);

            if (exposure > 0f)
            {
                cmd.SetGlobalMatrix(NonJitteredVP, viewProjection);
                cmd.SetGlobalFloat(HasLastPositionData, 0f);
                cmd.SetGlobalFloat(ForceNoMotion, 0f);
                cmd.SetGlobalFloat(MotionVectorDepthBias, 0f);

                // Background: every pixel reprojected from depth as part of the main body's frame.
                UpdateQuad(projection);
                cmd.SetGlobalMatrix(PreviousVP, pastBackgroundVP);
                cmd.DrawMesh(m_Quad, Matrix4x4.TRS(position, rotation, Vector3.one), Material, 0, 1);

                // Vessels: their own motion overwrites the background where they are the visible surface.
                cmd.SetGlobalMatrix(PreviousVP, pastObjectVP);
                m_Draws = DrawVessels(cmd, camera, exposure);

                if (TexturesUnlimitedFXLoader.defaultConfiguration.MotionVectorCheck)
                    Check(context, cmd, target, exposure, viewProjection, pastObjectVP, pastBackgroundVP, cameraVelocity);
            }
            cmd.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
            cmd.EndSample("TUFX Motion Vectors");

            m_Watch.Stop();
            m_BuildMs += m_Watch.Elapsed.TotalMilliseconds;
            ++m_BuildFrames;
        }

        static Matrix4x4 WorldToCamera(Vector3 position, Quaternion rotation)
        {
            return Matrix4x4.TRS(position, rotation, new Vector3(1f, 1f, -1f)).inverse;
        }

        static Quaternion PastRotation(Quaternion rotation, Vector3 angularVelocity, float dt)
        {
            float angle = angularVelocity.magnitude * dt * Mathf.Rad2Deg;
            if (angle < 1e-5f) return rotation;
            return Quaternion.AngleAxis(-angle, angularVelocity) * rotation;
        }

        // Past model transform of a rigid body: translated back by v*dt and rotated back by w*dt about its centre of mass.
        static Matrix4x4 PastBodyTransform(Vector3 velocity, Vector3 angularVelocity, Vector3 centre, float dt)
        {
            float angle = angularVelocity.magnitude * dt * Mathf.Rad2Deg;
            Quaternion back = angle < 1e-5f ? Quaternion.identity : Quaternion.AngleAxis(-angle, angularVelocity);
            return Matrix4x4.TRS(centre - velocity * dt, back, Vector3.one) * Matrix4x4.Translate(-centre);
        }

        // Camera motion is sampled once per physics step: KSP moves vessels (and CameraTools' dogfight camera) only in
        // FixedUpdate, so per-frame differences alternate between a step's motion and none. The estimate is the motion
        // over the last physics interval, the same instant the rigidbody velocities describe.
        void UpdateCameraMotion(Camera camera)
        {
            int frame = Time.frameCount;
            bool gap = frame - m_LastFrame > 2;   // not rendered for a while (effect off, scene change): start over
            if (frame == m_LastFrame) return;
            m_LastFrame = frame;
            if (gap) m_HasSample = false;
            double fixedTime = Time.fixedTime;
            if (m_HasSample && fixedTime == m_Sample.time) return;   // no physics step since the last sample

            var body = FlightGlobals.currentMainBody;
            Vector3d origin = body != null ? body.position : Vector3d.zero;
            var transform = camera.transform;
            var sample = new Sample
            {
                time = fixedTime,
                position = (Vector3d)transform.position - origin,
                rotation = transform.rotation,
            };

            bool valid = m_HasSample;
            double step = sample.time - m_Sample.time;
            if (valid && step > 0 && step < 0.25)
            {
                Vector3 velocity = (Vector3)((sample.position - m_Sample.position) / step);
                (sample.rotation * Quaternion.Inverse(m_Sample.rotation)).ToAngleAxis(out float angle, out Vector3 axis);
                if (angle > 180f) angle -= 360f;
                float angularSpeed = Mathf.Abs(angle) / (float)step;
                if (velocity.magnitude > CutSpeed || angularSpeed > CutAngularSpeed)
                    valid = false;   // camera cut
                else
                {
                    m_VelocityBody = velocity;
                    m_AngularVelocity = float.IsInfinity(axis.x) || Mathf.Abs(angle) < 1e-4f ? Vector3.zero : axis * (angle * Mathf.Deg2Rad / (float)step);
                }
            }
            else valid = false;

            if (!valid)
            {
                // No usable history (start, cut, long gap): assume the camera moves with the active vessel.
                var vessel = FlightGlobals.ActiveVessel;
                var rb = vessel != null && vessel.rootPart != null ? vessel.rootPart.Rigidbody : null;
                m_VelocityBody = rb != null ? rb.velocity + Krakensbane.GetFrameVelocityV3f() : Vector3.zero;
                m_AngularVelocity = Vector3.zero;
            }
            m_Sample = sample;
            m_HasSample = true;
        }

        void UpdateQuad(Matrix4x4 projection)
        {
            // Slightly larger than the frustum so jitter never uncovers an edge.
            var tan = new Vector2(1.02f / projection.m00, 1.02f / projection.m11);
            if (m_Quad != null && tan == m_QuadTan) return;
            m_QuadTan = tan;
            if (m_Quad == null)
                m_Quad = new Mesh { name = "TUFX motion background", hideFlags = HideFlags.DontSave };
            float x = tan.x * QuadDepth, y = tan.y * QuadDepth;
            m_QuadVerts[0] = new Vector3(-x, -y, QuadDepth);
            m_QuadVerts[1] = new Vector3(-x, y, QuadDepth);
            m_QuadVerts[2] = new Vector3(x, y, QuadDepth);
            m_QuadVerts[3] = new Vector3(x, -y, QuadDepth);
            m_Quad.vertices = m_QuadVerts;
            m_Quad.normals = m_QuadVerts;   // camera-space rays; the pass scales them to the far plane
            m_Quad.triangles = new[] { 0, 1, 2, 0, 2, 3 };
            m_Quad.bounds = new Bounds(Vector3.zero, Vector3.one * 1e6f);
        }

        int DrawVessels(CommandBuffer cmd, Camera camera, float exposure)
        {
            var vessels = FlightGlobals.VesselsLoaded;
            if (vessels == null) return 0;
            GeometryUtility.CalculateFrustumPlanes(camera, m_Planes);
            var material = Material;
            int cullingMask = camera.cullingMask;
            float now = Time.unscaledTime;
            int draws = 0;
            m_SkinnedDraws = 0;
            Vector3 cameraPosition = camera.transform.position;
            float pixelsPerRadian = camera.pixelHeight / (2f * Mathf.Tan(camera.fieldOfView * 0.5f * Mathf.Deg2Rad));

            for (int v = 0; v < vessels.Count; ++v)
            {
                var vessel = vessels[v];
                if (vessel == null || vessel.packed || vessel.state == Vessel.State.DEAD) continue;

                float radius = Mathf.Max(vessel.vesselSize.magnitude * 0.5f, 2f);
                if (!GeometryUtility.TestPlanesAABB(m_Planes, new Bounds(vessel.CoM, Vector3.one * (2f * radius)))) continue;

                if (!m_Vessels.TryGetValue(vessel, out var entry))
                    m_Vessels.Add(vessel, entry = new VesselEntry());
                if (entry.partCount != vessel.parts.Count || now - entry.builtAt > CacheRefresh)
                    Build(entry, vessel, cullingMask, now);

                // One rigid motion per vessel (root part's rigidbody): joint flex is far below a pixel over an exposure.
                Rigidbody rb = vessel.rootPart != null ? vessel.rootPart.Rigidbody : null;
                if (rb == null) continue;
                Matrix4x4 past = PastBodyTransform(rb.velocity, rb.angularVelocity, rb.worldCenterOfMass, exposure);

                // Renderers under MinPixels on screen keep the background motion.
                float distance = Mathf.Max((vessel.CoM - cameraPosition).magnitude - radius, camera.nearClipPlane);
                float minRadius = MinPixels * distance / pixelsPerRadian;
                var renderers = entry.renderers;
                for (int r = 0; r < renderers.Length; ++r)
                {
                    if (entry.radius[r] < minRadius) continue;
                    var renderer = renderers[r];
                    if (renderer == null || !renderer.isVisible) continue;
                    cmd.SetGlobalMatrix(PreviousM, past * renderer.localToWorldMatrix);
                    var subs = entry.submeshes[r];
                    for (int s = 0; s < subs.Length; ++s)
                        cmd.DrawRenderer(renderer, material, subs[s], 0);
                    draws += subs.Length;
                    if (renderer is SkinnedMeshRenderer) m_SkinnedDraws += subs.Length;
                }
            }

            if (m_Vessels.Count > vessels.Count)
            {
                foreach (var kv in m_Vessels)
                    if (kv.Key == null || !kv.Key.loaded) m_Stale.Add(kv.Key);
                foreach (var key in m_Stale) m_Vessels.Remove(key);
                m_Stale.Clear();
            }
            return draws;
        }

        void Build(VesselEntry entry, Vessel vessel, int cullingMask, float now)
        {
            entry.partCount = vessel.parts.Count;
            entry.builtAt = now;
            var renderers = new List<Renderer>();
            var submeshes = new List<int[]>();
            for (int i = 0; i < vessel.parts.Count; ++i)
            {
                var part = vessel.parts[i];
                if (part == null) continue;
                foreach (var mr in part.FindModelComponents<MeshRenderer>())
                {
                    var filter = mr.GetComponent<MeshFilter>();
                    Add(mr, filter != null ? filter.sharedMesh : null, cullingMask, renderers, submeshes);
                }
                foreach (var smr in part.FindModelComponents<SkinnedMeshRenderer>())
                    Add(smr, smr.sharedMesh, cullingMask, renderers, submeshes);
            }
            entry.renderers = renderers.ToArray();
            entry.submeshes = submeshes.ToArray();
            entry.radius = new float[entry.renderers.Length];
            for (int i = 0; i < entry.renderers.Length; ++i)
                entry.radius[i] = entry.renderers[i].bounds.extents.magnitude;
        }

        void Add(Renderer renderer, Mesh mesh, int cullingMask, List<Renderer> renderers, List<int[]> submeshes)
        {
            if (renderer == null || mesh == null) return;
            if (renderer.shadowCastingMode == ShadowCastingMode.ShadowsOnly) return;
            if ((cullingMask & (1 << renderer.gameObject.layer)) == 0) return;
            renderer.GetSharedMaterials(m_MatBuf);
            int count = Mathf.Min(m_MatBuf.Count, mesh.subMeshCount);
            m_SubBuf.Clear();
            for (int i = 0; i < count; ++i)
            {
                var material = m_MatBuf[i];
                if (material != null && material.renderQueue <= (int)RenderQueue.GeometryLast)
                    m_SubBuf.Add(i);
            }
            m_MatBuf.Clear();
            if (m_SubBuf.Count == 0) return;
            renderers.Add(renderer);
            submeshes.Add(m_SubBuf.ToArray());
        }

        // Self-test: once a second, compare the generated motion at each visible vessel's CoM and at the
        // sea-level point under the screen centre with the analytic displacement over the exposure.
        void Check(PostProcessRenderContext context, CommandBuffer cmd, int target, float exposure, Matrix4x4 viewProjection,
            Matrix4x4 pastObjectVP, Matrix4x4 pastBackgroundVP, Vector3 cameraVelocity)
        {
            if (m_CheckPending || Time.unscaledTime < m_NextCheck) return;
            m_NextCheck = Time.unscaledTime + 1f;
            int width = context.width, height = context.height;
            if (m_CheckRT == null || m_CheckRT.width != width || m_CheckRT.height != height)
            {
                if (m_CheckRT != null) m_CheckRT.Release();
                m_CheckRT = new RenderTexture(width, height, 0, RenderTextureFormat.RGHalf, RenderTextureReadWrite.Linear) { hideFlags = HideFlags.DontSave };
                m_CheckRT.Create();
            }

            m_CheckPoints.Clear();
            var camera = context.camera;
            foreach (var vessel in FlightGlobals.VesselsLoaded)
            {
                if (vessel == null || vessel.packed || vessel.rootPart == null) continue;
                var rb = vessel.rootPart.Rigidbody;
                if (rb == null) continue;
                Vector3 p = rb.worldCenterOfMass;
                Vector3 past = PastBodyTransform(rb.velocity, rb.angularVelocity, rb.worldCenterOfMass, exposure).MultiplyPoint(p);
                AddCheckPoint(vessel.vesselName, viewProjection, pastObjectVP, p, past, width, height);
            }
            var body = FlightGlobals.currentMainBody;
            if (body != null)
            {
                var ray = new Ray(camera.transform.position, camera.transform.forward);
                Vector3d oc = (Vector3d)ray.origin - body.position;
                double b = Vector3d.Dot(oc, (Vector3d)ray.direction);
                double c = oc.sqrMagnitude - body.Radius * body.Radius;
                double disc = b * b - c;
                if (disc > 0 && -b - Math.Sqrt(disc) > 0)
                {
                    Vector3 hit = ray.GetPoint((float)(-b - Math.Sqrt(disc)));
                    AddCheckPoint("sea level at centre", viewProjection, pastBackgroundVP, hit, hit, width, height);
                }
            }
            cmd.CopyTexture(target, m_CheckRT);
            m_CheckPending = true;
            float far = camera.farClipPlane;
            var points = m_CheckPoints.ToArray();
            float ms = m_BuildFrames > 0 ? (float)(m_BuildMs / m_BuildFrames) : 0f;
            int draws = m_Draws, skinned = m_SkinnedDraws;
            m_BuildMs = 0; m_BuildFrames = 0;
            Vector3 frameVelocity = Krakensbane.GetFrameVelocityV3f();
            Vector3 bodyVelocity = m_VelocityBody;
            Vector3 angular = m_AngularVelocity;
            cmd.RequestAsyncReadback(m_CheckRT, request =>
            {
                m_CheckPending = false;
                if (request.hasError) { Debug.Log("[TUFX] MotionVectorCheck: readback failed"); return; }
                var data = request.GetData<ushort>();
                // Infinite/NaN texels (half exponent all ones); their bounds locate the source.
                int bad = 0, minX = width, minY = height, maxX = -1, maxY = -1;
                for (int row = 0, i = 0; row < height; ++row)
                    for (int x = 0; x < width; ++x, i += 2)
                        if ((data[i] & 0x7C00) == 0x7C00 || (data[i + 1] & 0x7C00) == 0x7C00)
                        {
                            ++bad;
                            int y = height - 1 - row;
                            if (x < minX) minX = x;
                            if (x > maxX) maxX = x;
                            if (y < minY) minY = y;
                            if (y > maxY) maxY = y;
                        }
                var sb = new System.Text.StringBuilder();
                sb.AppendFormat("[TUFX] MotionVectorCheck exposure={0:F4}s build={1:F3}ms draws={2} skinned={3} camVelBody={4:F1}m/s camAng={5:F1}deg/s frameVel={6:F1}m/s camVelUnity={7:F1}m/s far={8:F0}m nonFinite={9}",
                    exposure, ms, draws, skinned, bodyVelocity.magnitude, angular.magnitude * Mathf.Rad2Deg, frameVelocity.magnitude, cameraVelocity.magnitude, far, bad);
                if (bad > 0)
                    sb.AppendFormat(" in px ({0},{1})-({2},{3})", minX, minY, maxX, maxY);
                foreach (var pt in points)
                {
                    // Readback rows run bottom-up; pixel.y counts from the top.
                    int x = Mathf.Clamp((int)pt.pixel.x, 0, width - 1);
                    int row = height - 1 - Mathf.Clamp((int)pt.pixel.y, 0, height - 1);
                    int i = (row * width + x) * 2;
                    var measured = new Vector2(Mathf.HalfToFloat(data[i]) * width, Mathf.HalfToFloat(data[i + 1]) * height);
                    sb.AppendFormat("\n  {0}: px=({1:F0},{2:F0}) expected=({3:F2},{4:F2}) measured=({5:F2},{6:F2})",
                        pt.label, pt.pixel.x, pt.pixel.y, pt.expected.x, pt.expected.y, measured.x, measured.y);
                }
                Debug.Log(sb.ToString());
            });
        }

        // Self-test of the blur input: counts NaN, infinite and very bright texels in the frame entering the reconstruction.
        public void CheckColor(CommandBuffer cmd, RenderTargetIdentifier source, int width, int height)
        {
            if (m_ColorCheckPending || Time.unscaledTime < m_NextColorCheck) return;
            m_NextColorCheck = Time.unscaledTime + 1f;
            if (m_ColorRT == null || m_ColorRT.width != width || m_ColorRT.height != height)
            {
                if (m_ColorRT != null) m_ColorRT.Release();
                m_ColorRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBHalf, RenderTextureReadWrite.Linear) { hideFlags = HideFlags.DontSave };
                m_ColorRT.Create();
            }
            cmd.Blit(source, m_ColorRT);
            m_ColorCheckPending = true;
            cmd.RequestAsyncReadback(m_ColorRT, request =>
            {
                m_ColorCheckPending = false;
                if (request.hasError) { Debug.Log("[TUFX] MotionBlurColorCheck: readback failed"); return; }
                var data = request.GetData<ushort>();
                int nan = 0, inf = 0, bright = 0;
                float max = 0f;
                for (int i = 0; i < data.Length; i += 4)
                {
                    bool isNan = false, isInf = false;
                    for (int c = 0; c < 3; ++c)
                    {
                        ushort h = data[i + c];
                        if ((h & 0x7C00) == 0x7C00) { if ((h & 0x03FF) != 0) isNan = true; else isInf = true; }
                        else max = Mathf.Max(max, Mathf.HalfToFloat(h));
                    }
                    if (isNan) ++nan;
                    else if (isInf) ++inf;
                    else if (Mathf.Max(Mathf.HalfToFloat(data[i]), Mathf.Max(Mathf.HalfToFloat(data[i + 1]), Mathf.HalfToFloat(data[i + 2]))) > 1000f) ++bright;
                }
                Debug.Log($"[TUFX] MotionBlurColorCheck nan={nan} inf={inf} over1000={bright} maxFinite={max:F0}");
            });
        }

        void AddCheckPoint(string label, Matrix4x4 vp, Matrix4x4 pastVP, Vector3 now, Vector3 past, int width, int height)
        {
            Vector4 c = vp * new Vector4(now.x, now.y, now.z, 1f);
            if (c.w <= 0f) return;
            Vector2 uv = new Vector2(c.x / c.w * 0.5f + 0.5f, c.y / c.w * 0.5f + 0.5f);
            if (uv.x < 0f || uv.x > 1f || uv.y < 0f || uv.y > 1f) return;
            Vector4 o = pastVP * new Vector4(past.x, past.y, past.z, 1f);
            if (o.w <= 0f) return;
            Vector2 uvOld = new Vector2(o.x / o.w * 0.5f + 0.5f, o.y / o.w * 0.5f + 0.5f);
            // Texture rows run top-down on D3D, as in the shader's UNITY_UV_STARTS_AT_TOP branch.
            bool flip = SystemInfo.graphicsUVStartsAtTop;
            Vector2 d = uv - uvOld;
            if (flip) { d.y = -d.y; uv.y = 1f - uv.y; }
            m_CheckPoints.Add(new CheckPoint
            {
                label = label,
                pixel = new Vector2(uv.x * width, uv.y * height),
                expected = new Vector2(d.x * width, d.y * height),
            });
        }
    }
}
