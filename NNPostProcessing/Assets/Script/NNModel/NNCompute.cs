using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NNPP
{
    public class NNCompute
    {
        private static NNCompute _instance;

        public static NNCompute Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new NNCompute();
                    _instance.Init();
                }

                return _instance;
            }

        }

        private int[] Conv2DKernelLayers = new int[6] { 8, 12, 16, 20, 32, 64 };
        private int[] Conv2DKernels = new int[6];

        public ComputeShader Shader;
        private string shaderpath = "NNLayer";
        private int LeakyReluKernel, BatchNormalizationKernel, InputLayerKernel, AddKernel,
            ConcatenateKernel, OutputLayerKernel, UpSampling2DKernel, ReluKernel, TanhKernel;

        private void Init()
        {
            Conv2DKernels = new int[6];
            Shader = Resources.Load<ComputeShader>(shaderpath);
            for (int i = 0; i < Conv2DKernelLayers.Length; i++)
            {
                Conv2DKernels[i] = Shader.FindKernel(string.Format("Conv2D_{0}", Conv2DKernelLayers[i]));
            }
            LeakyReluKernel = Shader.FindKernel("LeakyReLU");
            BatchNormalizationKernel = Shader.FindKernel("BatchNormalization");
            InputLayerKernel = Shader.FindKernel("InputLayer");
            OutputLayerKernel = Shader.FindKernel("OutputLayer");
            UpSampling2DKernel = Shader.FindKernel("UpSampling2D");
            ConcatenateKernel = Shader.FindKernel("Concatenate");
            ReluKernel = Shader.FindKernel("ReLU");
            TanhKernel = Shader.FindKernel("Tanh");
            AddKernel = Shader.FindKernel("Add");
        }

        public int KernelConv2D(int channel)
        {
            for (int i = 0; i < Conv2DKernelLayers.Length; i++)
            {
                if (channel <= Conv2DKernelLayers[i])
                {
                    return Conv2DKernels[i];
                }
            }
            return -1;
        }

        public int Kernel(string name)
        {
            switch (name)
            {
                case ("LeakyReLU"):
                    return LeakyReluKernel;
                case ("BatchNormalization"):
                    return BatchNormalizationKernel;
                case ("InputLayer"):
                    return InputLayerKernel;
                case ("OutputLayer"):
                    return OutputLayerKernel;
                case ("UpSampling2D"):
                    return UpSampling2DKernel;
                case ("ReLU"):
                    return ReluKernel;
                case ("Tanh"):
                    return TanhKernel;
                case ("Concatenate"):
                    return ConcatenateKernel;
                case ("Add"):
                    return AddKernel;
                default:
                    return -1;
            }
        }
    }
}