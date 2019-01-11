// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{

    [System.Serializable]
    [RequireComponent(typeof(Camera))]
    public class NNPostProcessingEffect : MonoBehaviour
    {
        public NNStyle style = NNStyle.starry_night;

        private NNModel model;

        void Start()
        {
            model = new NNModel();
            model.Load(style.ToString());
        }

        void OnDisable()
        {
            model.Release();
        }

        void OnRenderImage(RenderTexture src, RenderTexture dst)
        {
            var predict = model.Predict(src);
            Graphics.Blit(predict, dst);
        }

    }

    public enum NNStyle
    {
        des_glaneuses,
        la_muse,
        mirror,
        sketch,
        starry_night,
        udnie,
        wave_crop,
        gritty,
        watercolor
    }
}