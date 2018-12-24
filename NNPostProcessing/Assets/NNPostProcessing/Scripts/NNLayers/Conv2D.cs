// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    [System.Serializable]
    public class Conv2D : NNLayerBase
    {
        public int Filters;
        public Vector2Int KernalSize;
        public Vector2Int Stride;
        private ComputeBuffer outputbuffer;
        private ComputeBuffer weightbuffer;
        public Conv2D(KerasLayerConfigJson config) : base(config)
        {
            Filters = config.filters;
            KernalSize = new Vector2Int(config.kernel_size[0], config.kernel_size[1]);
            Stride = new Vector2Int(config.strides[0], config.strides[1]);
            KernelId = NNCompute.Instance.KernelConv2D(32);
        }

        public override void LoadWeight(KerasLayerWeightJson[] weightsKernel)
        {
            WeightShape = new Vector4(weightsKernel[0].shape[0],
                weightsKernel[0].shape[1],
                weightsKernel[0].shape[2],
                weightsKernel[0].shape[3]);
            int kernel_weight_length = (int)(WeightShape.x * WeightShape.y * WeightShape.z * WeightShape.w);
            int bias_weight_length = (int)WeightShape.w;
            float[] Weights = new float[kernel_weight_length + bias_weight_length];
            for (int i = 0; i < WeightShape.x; i++)
            {
                for (int j = 0; j < WeightShape.y; j++)
                {
                    for (int k = 0; k < WeightShape.z; k++)
                    {
                        for (int w = 0; w < WeightShape.w; w++)
                        {
                            float arrayindex = i * WeightShape.y * WeightShape.z * WeightShape.w +
                                             j * WeightShape.z * WeightShape.w +
                                             k * WeightShape.w +
                                             w;
                            Weights[(int)arrayindex] = weightsKernel[0].kernelweight[i, j, k, w];
                        }
                    }
                }
            }
            Array.Copy(weightsKernel[1].arrayweight, 0, Weights, kernel_weight_length, bias_weight_length);
            if (weightbuffer != null)
                weightbuffer.Release();
            weightbuffer = new ComputeBuffer(kernel_weight_length + bias_weight_length, sizeof(float));
            weightbuffer.SetData(Weights);
        }

        public override void Init(Vector3Int inputShape)
        {
            InputShape = inputShape;
            OutputShape = new Vector3Int(inputShape.x / Stride.x, inputShape.y / Stride.y, Filters);
            if (outputbuffer != null)
                outputbuffer.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            int maxfilter = Mathf.Max(inputShape.z, Filters);
            KernelId = NNCompute.Instance.KernelConv2D(maxfilter);
            Output = outputbuffer;
        }

        public override void Release()
        {
            if (weightbuffer != null)
                weightbuffer.Release();
            if (outputbuffer != null)
                outputbuffer.Release();
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
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "InputShapeIdMultiplier", new int[3]
            {
                InputShape.y * InputShape.z,
                InputShape.z,
                1
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "OutputShape", new int[3]
            {
                OutputShape.x,
                OutputShape.y,
                OutputShape.z
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "OutputShapeIdMultiplier", new int[3]
            {
                OutputShape.y * OutputShape.z,
                OutputShape.z,
                1
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "WeightsShape", new int[4]
            {
                KernalSize.x,
                KernalSize.y,
                InputShape.z,
                Filters
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "WeightsShapeIdMultiplier", new int[4]
            {
                KernalSize.y * InputShape.z * Filters,
                InputShape.z * Filters,
                Filters,
                1
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "Stride", new int[2]
            {
                Stride.x,
                Stride.y
            });
            cmd.DispatchCompute(NNCompute.Instance.Shader, KernelId, Mathf.CeilToInt(OutputShape.x / 4.0f), OutputShape.y, 1);
        }
    }
}