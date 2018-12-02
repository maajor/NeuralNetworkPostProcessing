using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;
using Newtonsoft.Json;

public class ModelParser : MonoBehaviour {
    

#if UNITY_EDITOR
    [MenuItem("Assets/TestPredict")]
    private static void Predict()
    {
        /*Texture2D input = Selection.activeObject as Texture2D;
        var model = new NeuralNetworkModel();
        model.Load();
        model.Init(input);
        DateTime start = DateTime.Now;
        Texture output = model.Predict(input);
        DateTime end = DateTime.Now;
        TimeSpan span = end - start;
        Debug.Log(span.Milliseconds + "ms");
        if (output != null)
        {
            SaveRTToFile(output as RenderTexture);
        }
        model.Release();*/
    }

    public static void SaveRTToFile(RenderTexture rt)
    {
        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        RenderTexture.active = null;

        byte[] bytes;
        bytes = tex.EncodeToPNG();

        string path = Application.dataPath + "/Art/out.png";
        System.IO.File.WriteAllBytes(path, bytes);
        AssetDatabase.ImportAsset(path);
        Debug.Log("Saved to " + path);
    }
#endif
}

[System.Serializable]
public class KerasJson
{
    public KerasModelJson model;
    public List<KerasLayerWeightJson> weights;
}

[System.Serializable]
public class KerasLayerWeightJson
{
    public int[] shape;
    public float[] arrayweight;
    public float[,,,] kernelweight;
}

[System.Serializable]
public class KerasModelJson
{
    public string class_name;
    public KerasLayersJson config;
}
[System.Serializable]
public class KerasLayersJson
{
    public string name;
    public KerasLayerJson[] layers;
}
[System.Serializable]
public class KerasLayerJson
{
    public string name;
    public string class_name;
    public KerasLayerConfigJson config;
    public List<List<List<object>>> inbound_nodes;
}
[System.Serializable]
public class KerasLayerConfigJson
{
    public string name;
    public int filters;
    public int[] kernel_size;
    public int[] strides;
    public int[] size;
    public float alpha;
    public float momentum;
    public string activation;
}