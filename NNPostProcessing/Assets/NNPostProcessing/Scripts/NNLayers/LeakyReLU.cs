// neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class LeakyReLU : ReLU
    {
        public float Alpha;
        public LeakyReLU() : base()
        {
            KernelId = NNCompute.Instance.Kernel("LeakyReLU");
        }

        public override void Run(object[] input)
        {
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[0] as ComputeBuffer);
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);
            NNCompute.Instance.Shader.SetFloat("Alpha", Alpha);
            NNCompute.Instance.Shader.Dispatch(KernelId, Mathf.CeilToInt(OutputShape.x * OutputShape.y * OutputShape.z / 32.0f), 1, 1);
        }
    }
}