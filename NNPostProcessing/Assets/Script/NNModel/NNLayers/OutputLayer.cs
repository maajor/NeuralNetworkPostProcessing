﻿using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class OutputLayer : NNLayerBase
    {
        public RenderTexture outputTex;
        public OutputLayer(KerasLayerConfigJson config) : base(config)
        {
            KernelId = NNCompute.Instance.Kernel("OutputLayer");
        }

        public override void Run(object[] input, CommandBuffer cmd)
        {
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
            cmd.SetComputeTextureParam(NNCompute.Instance.Shader, KernelId, "OutputImage", outputTex);
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "InputShape", new int[3]
            {
                InputShape.x,
                InputShape.y,
                InputShape.z
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "InputShapeIdMultiplier", new int[3]
            {
                InputShape.y * InputShape.z,
                InputShape.z,
                1
            });
            cmd.DispatchCompute(NNCompute.Instance.Shader, KernelId, OutputShape.x / 8, OutputShape.y / 8, 1);
        }

        public override void Init(int4 inputShape)
        {
            base.Init(inputShape);
            outputTex?.Release();
            outputTex = new RenderTexture(OutputShape.y, OutputShape.x, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            outputTex.enableRandomWrite = true;
            outputTex.Create();
        }

        public override void Release()
        {
            outputTex?.Release();
        }
    }
}