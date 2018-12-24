using System.Collections;
using System.Collections.Generic;
using NNPP;
using UnityEngine;
using UnityEditor;
using UnityEngine.Rendering;

public class NNDebug : MonoBehaviour {

    [MenuItem("Assets/test")]
    public static void debug()
    {
        Texture tex = Selection.activeObject as Texture;
        Debug.Log(tex);

        RenderTargetIdentifier source = new RenderTargetIdentifier(tex);
        CommandBuffer cb = new CommandBuffer();
        cb.Blit(tex, source);

        for (int i = 1; i < 32; i++)
        {
            NNModel model = new NNModel();
            model.debug_layer = i;
            model.Load("starry_night");
            model.Setup(cb, source, BuiltinRenderTextureType.ResolvedDepth, tex.height, tex.width);
            var dst = model.Predict();

            Camera.main.AddCommandBuffer(CameraEvent.AfterEverything, cb);
            Camera.main.Render();

            SaveRTToFile(dst, i);

            model.Release();
        }
    }

    public static void SaveRTToFile(RenderTexture rt, int id)
    {
        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        RenderTexture.active = null;

        byte[] bytes;
        bytes = tex.EncodeToPNG();

        string path = Application.dataPath + "/Art/out_" + id + ".png";
        System.IO.File.WriteAllBytes(path, bytes);
        AssetDatabase.ImportAsset(path);
        Debug.Log("Saved to " + path);
    }
}
