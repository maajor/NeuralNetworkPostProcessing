using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.PostProcessing;

[System.Serializable]
[PostProcess(typeof(NNPostProcessingRenderer), PostProcessEvent.BeforeStack, "NNPP")]
public sealed class NNPostProcessingEffect : PostProcessEffectSettings
{
}

public class NNPostProcessingRenderer : PostProcessEffectRenderer<NNPostProcessingEffect>
{

    private NeuralNetworkModel model;

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
        model = new NeuralNetworkModel();
        model.Load();
    }

    public override void Release()
    {
        base.Release();
        model.Release();
    }
}
