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
    }
}