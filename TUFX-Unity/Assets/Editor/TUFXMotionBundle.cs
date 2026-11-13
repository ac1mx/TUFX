using System;
using System.IO;
using UnityEditor;

// Builds GameData/TUFX/Shaders/tufx-motion.ssf (TUFX's guarded motion blur shader).
// Batch: Unity.exe -batchmode -quit -projectPath <project> -executeMethod TUFXMotionBundle.Build -outputPath <dir>
public static class TUFXMotionBundle
{
    const string BundleName = "tufx-motion.ssf";
    const string ShaderPath = "Assets/Shaders/MotionBlur/TUFX-MotionBlur.shader";

    [MenuItem("Assets/Build TUFX Motion Blur Bundle Win64")]
    public static void Build()
    {
        const string staging = "Assets/Build";
        Directory.CreateDirectory(staging);
        var build = new AssetBundleBuild { assetBundleName = BundleName, assetNames = new[] { ShaderPath } };
        var manifest = BuildPipeline.BuildAssetBundles(staging, new[] { build }, BuildAssetBundleOptions.ForceRebuildAssetBundle, BuildTarget.StandaloneWindows64);
        if (manifest == null)
            throw new Exception("TUFXMotionBundle: build failed");
        string output = ArgValue("-outputPath");
        if (output != null)
        {
            Directory.CreateDirectory(output);
            File.Copy(Path.Combine(staging, BundleName), Path.Combine(output, BundleName), true);
        }
    }

    static string ArgValue(string name)
    {
        var args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; ++i)
            if (args[i] == name) return args[i + 1];
        return null;
    }
}
