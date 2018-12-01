using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NNPostProcessing : MonoBehaviour
{

    private NeuralNetworkModel model;
    private int height, width;
    void Start()
    {
        model = new NeuralNetworkModel();
        model.Load();
        height = 0;
        width = 0;
    }

    void OnDisable()
    {
        model.Release();
    }

    void OnRenderImage(RenderTexture src, RenderTexture dst)
    {
        if (src == null)
            return;
        if (height != src.height || width != src.width)
        {
            model.Init(src);
            height = src.height;
            width = src.width;
        }
        var dsttex = model.Predict(src);
        Graphics.Blit(dsttex, dst);
    }
}
