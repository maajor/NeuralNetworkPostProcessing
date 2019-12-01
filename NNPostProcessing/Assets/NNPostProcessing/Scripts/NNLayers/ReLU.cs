// neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class ReLU : NNLayerBase
    {
        protected ComputeBuffer outputbuffer;
        public ReLU() : base()
        {
            KernelId = NNCompute.Instance.Kernel("ReLU");
        }

        public override void Init(Vector3Int inputShape)
        {
            base.Init(inputShape);
            if (outputbuffer != null)
                outputbuffer.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            Output = outputbuffer;
        }

        public override void Release()
        {
            if (outputbuffer != null)
                outputbuffer.Release();
        }

        public override void Run(object[] input)
        {
            //Unity2019 refuse threadgroup > 65536, so we make a 2d input threadgroup
            int threadGroupX = Mathf.CeilToInt(OutputShape.y * OutputShape.z / 32.0f);
            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier", new int[3]
            {
                OutputShape.x,
                1,
                1
            });
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[0] as ComputeBuffer);
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);
            NNCompute.Instance.Shader.Dispatch(KernelId, threadGroupX, OutputShape.x, 1);
        }
    }
}