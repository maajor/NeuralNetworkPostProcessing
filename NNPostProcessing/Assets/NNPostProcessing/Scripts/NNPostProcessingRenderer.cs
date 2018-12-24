// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.PostProcessing;

namespace NNPP
{
    [System.Serializable]
    public class NNPostProcessingRenderer : PostProcessEffectRenderer<NNPostProcessingEffect>
    {

        private NNModel model;

        public override void Render(PostProcessRenderContext context)
        {
            if (Application.isPlaying)
            {

                var cmd = context.command;
                cmd.BeginSample("NNPP");
                model.Setup(cmd, context.source, BuiltinRenderTextureType.ResolvedDepth, context.screenHeight, context.screenWidth);
                var dst = model.Predict();
                cmd.BlitFullscreenTriangle(dst, context.destination);
                cmd.EndSample("NNPP");
            }
            else
            {
                context.command.BlitFullscreenTriangle(context.source, context.destination);
            }
        }

        public override void Init()
        {
            base.Init();
            model = new NNModel();
            model.Load(settings.style.value.ToString());
        }

        public override void Release()
        {
            base.Release();
            model.Release();
        }
    }
}