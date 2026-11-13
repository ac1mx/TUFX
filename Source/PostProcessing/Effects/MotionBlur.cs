using System;
using TUFX;

namespace UnityEngine.Rendering.PostProcessing
{
    /// <summary>
    /// This class holds settings for the Motion Blur effect.
    /// </summary>
    [Serializable]
    [PostProcess(typeof(MotionBlurRenderer), "Unity/Motion Blur", false)]
    public sealed class MotionBlur : PostProcessEffectSettings
    {
        /// <summary>
        /// TUFX's motion blur shader (tufx-motion.ssf): the stock passes with NaN/infinity guards.
        /// </summary>
        internal const string GuardedShaderName = "Hidden/TUFX/MotionBlur";

        /// <summary>
        /// The angle of the rotary shutter. Larger values give longer exposure therefore a stronger
        /// blur effect.
        /// </summary>
        [Range(0f, 360f), Tooltip("The angle of rotary shutter. Larger values give longer exposure.")]
        public FloatParameter shutterAngle = new FloatParameter { value = 270f };

        /// <summary>
        /// The amount of sample points, which affects quality and performances.
        /// </summary>
        [Range(4, 32), Tooltip("The amount of sample points. This affects quality and performance.")]
        public IntParameter sampleCount = new IntParameter { value = 10 };

        /// <summary>
        /// Frame rate the shutter angle refers to: exposure = shutterAngle / 360 / shutterFps seconds,
        /// independent of the render frame rate (24 = cinema). 0 = the render frame time (stock behaviour).
        /// </summary>
        [Range(0f, 120f), Tooltip("Frame rate the shutter angle refers to (24 = film). 0 = render frame rate.")]
        public FloatParameter shutterFps = new FloatParameter { value = 24f };

        /// <summary>
        /// Motion from vessel and camera velocities (flight camera) instead of Unity's per-frame motion vectors,
        /// which KSP only updates on physics steps.
        /// </summary>
        [Tooltip("Blur moving vessels from their velocities (flight camera).")]
        public BoolParameter objectMotion = new BoolParameter { value = true };

        /// <summary>
        /// Longest blur streak, in percent of the screen height.
        /// </summary>
        [Range(1f, 10f), Tooltip("Longest blur, percent of screen height.")]
        public FloatParameter maxBlurRadius = new FloatParameter { value = 5f };

        /// <summary>
        /// Returns <c>true</c> if the effect is currently enabled and supported.
        /// </summary>
        /// <param name="context">The current post-processing render context</param>
        /// <returns><c>true</c> if the effect is currently enabled and supported</returns>
        public override bool IsEnabledAndSupported(PostProcessRenderContext context)
        {
            return enabled.value
                && shutterAngle.value > 0f
            #if UNITY_EDITOR
                // Don't render motion blur preview when the editor is not playing as it can in some
                // cases results in ugly artifacts (i.e. when resizing the game view).
                && Application.isPlaying
            #endif
                && SystemInfo.supportsMotionVectors
                && RenderTextureFormat.RGHalf.IsSupported()
                && !context.stereoActive;
        }

        public override void Load(ConfigNode config)
        {
            loadFloatParameter(config, "ShutterAngle", shutterAngle);
            loadIntParameter(config, "SampleCount", sampleCount);
            loadFloatParameter(config, "ShutterFps", shutterFps);
            loadBoolParameter(config, "ObjectMotion", objectMotion);
            loadFloatParameter(config, "MaxBlurRadius", maxBlurRadius);
        }

        public override void Save(ConfigNode config)
        {
            saveFloatParameter(config, "ShutterAngle", shutterAngle);
            saveIntParameter(config, "SampleCount", sampleCount);
            saveFloatParameter(config, "ShutterFps", shutterFps);
            saveBoolParameter(config, "ObjectMotion", objectMotion);
            saveFloatParameter(config, "MaxBlurRadius", maxBlurRadius);
        }

        /// <summary>
        /// Exposure in game seconds for the current frame.
        /// </summary>
        internal float Exposure()
        {
            float fraction = shutterAngle.value / 360f;
            return shutterFps.value > 0f ? fraction / shutterFps.value * Time.timeScale : fraction * Time.deltaTime;
        }

    }
    [UnityEngine.Scripting.Preserve]
    internal sealed class MotionBlurRenderer : PostProcessEffectRenderer<MotionBlur>
    {
        enum Pass
        {
            VelocitySetup,
            TileMax1,
            TileMax2,
            TileMaxV,
            NeighborMax,
            Reconstruction
        }

        static readonly int CameraMotionVectorsTexture = Shader.PropertyToID("_CameraMotionVectorsTexture");
        static readonly int CameraMotionVectorsTextureTexelSize = Shader.PropertyToID("_CameraMotionVectorsTexture_TexelSize");
        static readonly int ExposureMotionTex = Shader.PropertyToID("_TUFXExposureMotionTex");

        ObjectMotionVectors m_Motion;

        static bool Guarded(PostProcessRenderContext context)
        {
            var shader = context.resources.shaders.motionBlur;
            return shader != null && shader.name == MotionBlur.GuardedShaderName;
        }

        // Object motion needs the guarded shader: exposure-long vectors reach the sun and ocean glints, whose
        // infinite colours the stock reconstruction spreads as NaN.
        bool UsesObjectMotion(PostProcessRenderContext context)
        {
            return settings.objectMotion.value && context.motionCommand != null && Guarded(context) && ObjectMotionVectors.Supported(context.camera);
        }

        public override DepthTextureMode GetCameraFlags()
        {
            return DepthTextureMode.Depth | DepthTextureMode.MotionVectors;
        }

        public override DepthTextureMode GetCameraFlags(PostProcessRenderContext context)
        {
            return UsesObjectMotion(context) ? DepthTextureMode.Depth : GetCameraFlags();
        }

        public override void Release()
        {
            m_Motion?.Release();
            m_Motion = null;
            base.Release();
        }

        private void CreateTemporaryRT(PostProcessRenderContext context, int nameID, int width, int height, RenderTextureFormat RTFormat)
        {
            var cmd = context.command;
            var rtDesc = context.GetDescriptor(0, RTFormat, RenderTextureReadWrite.Linear);
            rtDesc.width = width;
            rtDesc.height = height;
#if UNITY_2019_1_OR_NEWER
            cmd.GetTemporaryRT(nameID, rtDesc, FilterMode.Point);
#elif UNITY_2017_3_OR_NEWER
            cmd.GetTemporaryRT(nameID, rtDesc.width, rtDesc.height, rtDesc.depthBufferBits, FilterMode.Point, rtDesc.colorFormat, RenderTextureReadWrite.Linear, rtDesc.msaaSamples, rtDesc.enableRandomWrite, rtDesc.memoryless, context.camera.allowDynamicResolution);
#else
            cmd.GetTemporaryRT(nameID, rtDesc.width, rtDesc.height, rtDesc.depthBufferBits, FilterMode.Point, rtDesc.colorFormat, RenderTextureReadWrite.Linear, rtDesc.msaaSamples, rtDesc.enableRandomWrite, rtDesc.memoryless);
#endif
        }

        public override void Render(PostProcessRenderContext context)
        {
            var cmd = context.command;

            if (m_ResetHistory)
            {
                cmd.BlitFullscreenTriangle(context.source, context.destination);
                m_ResetHistory = false;
                return;
            }

            var vectorRTFormat = RenderTextureFormat.RGHalf;
            bool objectMotion = UsesObjectMotion(context);
            // Velocity + linear 0..1 depth. KSP's far plane is 750 km, so a 10-bit depth puts every craft and the sea
            // behind it at 0 and the reconstruction can no longer tell foreground from background; half floats keep
            // relative precision. Half floats also keep NaN and exact zeros, which only the guarded shader handles.
            var packedRTFormat = Guarded(context) && RenderTextureFormat.ARGBHalf.IsSupported()
                ? RenderTextureFormat.ARGBHalf
                : RenderTextureFormat.ARGB2101010.IsSupported() ? RenderTextureFormat.ARGB2101010 : RenderTextureFormat.ARGB32;

            var sheet = context.propertySheets.Get(context.resources.shaders.motionBlur);
            cmd.BeginSample("MotionBlur");

            // Calculate the maximum blur radius in pixels.
            int maxBlurPixels = Mathf.Max(1, (int)(settings.maxBlurRadius.value * context.height / 100));

            // Calculate the TileMax size.
            // It should be a multiple of 8 and larger than maxBlur.
            int tileSize = ((maxBlurPixels - 1) / 8 + 1) * 8;

            // Pass 1 - Velocity/depth packing
            float exposure = settings.Exposure();
            float velocityScale;
            if (objectMotion)
            {
                // Motion over the exposure, generated before the stack runs; replaces Unity's per-frame vectors for this pass.
                var motionCmd = context.motionCommand;
                motionCmd.GetTemporaryRT(ExposureMotionTex, context.width, context.height, 0, FilterMode.Point, vectorRTFormat, RenderTextureReadWrite.Linear);
                if (m_Motion == null) m_Motion = new ObjectMotionVectors();
                m_Motion.Render(context, motionCmd, ExposureMotionTex, exposure);
                cmd.SetGlobalTexture(CameraMotionVectorsTexture, ExposureMotionTex);
                cmd.SetGlobalVector(CameraMotionVectorsTextureTexelSize, new Vector4(1f / context.width, 1f / context.height, context.width, context.height));
                velocityScale = 1f;
            }
            else
            {
                // Unity's vectors span one frame; scale them to the exposure.
                velocityScale = Time.deltaTime > 0f ? exposure / Time.deltaTime : 0f;
            }
            sheet.properties.SetFloat(ShaderIDs.VelocityScale, velocityScale);
            sheet.properties.SetFloat(ShaderIDs.MaxBlurRadius, maxBlurPixels);
            sheet.properties.SetFloat(ShaderIDs.RcpMaxBlurRadius, 1f / maxBlurPixels);

            int vbuffer = ShaderIDs.VelocityTex;
            CreateTemporaryRT(context, vbuffer, context.width, context.height, packedRTFormat);
            cmd.BlitFullscreenTriangle(BuiltinRenderTextureType.None, vbuffer, sheet, (int)Pass.VelocitySetup);
            if (objectMotion)
            {
                cmd.ReleaseTemporaryRT(ExposureMotionTex);
                cmd.SetGlobalTexture(CameraMotionVectorsTexture, BuiltinRenderTextureType.MotionVectors);
            }

            // Pass 2 - First TileMax filter (1/2 downsize)
            int tile2 = ShaderIDs.Tile2RT;
            CreateTemporaryRT(context, tile2, context.width / 2, context.height / 2, vectorRTFormat);
            cmd.BlitFullscreenTriangle(vbuffer, tile2, sheet, (int)Pass.TileMax1);

            // Pass 3 - Second TileMax filter (1/2 downsize)
            int tile4 = ShaderIDs.Tile4RT;
            CreateTemporaryRT(context, tile4, context.width / 4, context.height / 4, vectorRTFormat);
            cmd.BlitFullscreenTriangle(tile2, tile4, sheet, (int)Pass.TileMax2);
            cmd.ReleaseTemporaryRT(tile2);

            // Pass 4 - Third TileMax filter (1/2 downsize)
            int tile8 = ShaderIDs.Tile8RT;
            CreateTemporaryRT(context, tile8, context.width / 8, context.height / 8, vectorRTFormat);
            cmd.BlitFullscreenTriangle(tile4, tile8, sheet, (int)Pass.TileMax2);
            cmd.ReleaseTemporaryRT(tile4);

            // Pass 5 - Fourth TileMax filter (reduce to tileSize)
            var tileMaxOffs = Vector2.one * (tileSize / 8f - 1f) * -0.5f;
            sheet.properties.SetVector(ShaderIDs.TileMaxOffs, tileMaxOffs);
            sheet.properties.SetFloat(ShaderIDs.TileMaxLoop, (int)(tileSize / 8f));

            int tile = ShaderIDs.TileVRT;
            CreateTemporaryRT(context, tile, context.width / tileSize, context.height / tileSize, vectorRTFormat);
            cmd.BlitFullscreenTriangle(tile8, tile, sheet, (int)Pass.TileMaxV);
            cmd.ReleaseTemporaryRT(tile8);

            // Pass 6 - NeighborMax filter
            int neighborMax = ShaderIDs.NeighborMaxTex;
            CreateTemporaryRT(context, neighborMax, context.width / tileSize, context.height / tileSize, vectorRTFormat);
            cmd.BlitFullscreenTriangle(tile, neighborMax, sheet, (int)Pass.NeighborMax);
            cmd.ReleaseTemporaryRT(tile);

            // Pass 7 - Reconstruction pass
            sheet.properties.SetFloat(ShaderIDs.LoopCount, Mathf.Clamp(settings.sampleCount / 2, 1, 64));
            if (m_Motion != null && TexturesUnlimitedFXLoader.defaultConfiguration.MotionVectorCheck)
                m_Motion.CheckColor(cmd, context.source, context.width, context.height);
            cmd.BlitFullscreenTriangle(context.source, context.destination, sheet, (int)Pass.Reconstruction);

            cmd.ReleaseTemporaryRT(vbuffer);
            cmd.ReleaseTemporaryRT(neighborMax);
            cmd.EndSample("MotionBlur");
        }
    }
}
