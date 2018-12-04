using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    [System.Serializable]
    public class BatchNormalization : NNLayerBase
    {
        private ComputeBuffer weightbuffer;
        private ComputeBuffer outputbuffer;
        public BatchNormalization(KerasLayerConfigJson config) : base(config)
        {
            KernelId = NNCompute.Instance.Kernel("BatchNormalization");
        }

        public override void LoadWeight(KerasLayerWeightJson[] weightsKernel)
        {
            WeightShape.x = weightsKernel[0].shape[0];
            float[] Weights = new float[WeightShape.x * 4];
            for (int i = 0; i < WeightShape.x; i++)
            {
                Weights[i * 4] = weightsKernel[0].arrayweight[i];
                Weights[i * 4 + 1] = weightsKernel[1].arrayweight[i];
                Weights[i * 4 + 2] = weightsKernel[2].arrayweight[i];
                Weights[i * 4 + 3] = weightsKernel[3].arrayweight[i];
            }
            weightbuffer?.Release();
            weightbuffer = new ComputeBuffer(WeightShape.x * 4, sizeof(float));
            weightbuffer.SetData(Weights);
        }

        public override void Init(int4 inputShape)
        {
            base.Init(inputShape);
            outputbuffer?.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            Output = outputbuffer;
        }

        public override void Release()
        {
            outputbuffer?.Release();
            weightbuffer?.Release();
        }

        public override void Run(object[] input, CommandBuffer cmd)
        {
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "Weights", weightbuffer);
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "InputShape", new int[3]
            {
                InputShape.x,
                InputShape.y,
                InputShape.z
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "OutputShape", new int[3]
            {
                OutputShape.x,
                OutputShape.y,
                OutputShape.z
            });
            cmd.DispatchCompute(NNCompute.Instance.Shader, KernelId, OutputShape.x * OutputShape.y / 32, OutputShape.z, 1);
        }
    }
}