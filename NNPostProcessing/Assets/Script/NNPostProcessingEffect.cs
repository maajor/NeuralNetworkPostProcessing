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
    [PostProcess(typeof(NNPostProcessingRenderer), PostProcessEvent.BeforeStack, "NNPP")]
    public sealed class NNPostProcessingEffect : PostProcessEffectSettings
    {
        [DisplayName("Type"), Tooltip("Neural Network Style Type")]
        public NNStyleParameter style = new NNStyleParameter { value = NNStyle.starry_night };

        public override bool IsEnabledAndSupported(PostProcessRenderContext context)
        {
            return enabled.value
                   && SystemInfo.supportsComputeShaders
                   && !RuntimeUtilities.isAndroidOpenGL;
        }
    }

    [System.Serializable]
    public sealed class NNStyleParameter : ParameterOverride<NNStyle> { }

    public enum NNStyle
    {
        des_glaneuses,
        la_muse,
        mirror,
        sketch,
        starry_night,
        udnie,
        wave_crop,
    }
}