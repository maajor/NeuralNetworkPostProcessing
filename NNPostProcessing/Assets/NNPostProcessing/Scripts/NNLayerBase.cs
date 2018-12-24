// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    [System.Serializable]
    public class NNLayerBase
    {
        public string Name;
        public Vector3Int InputShape;
        public Vector3Int OutputShape;
        public Vector4 WeightShape;
        public object Output;
        [SerializeField]
        protected int KernelId;
        public NNLayerBase(KerasLayerConfigJson config)
        {
            if (config != null)
                Name = config.name;
        }

        public virtual void LoadWeight(KerasLayerWeightJson[] weights)
        {

        }

        public virtual void Run(object[] input, CommandBuffer cmd)
        {
            Output = input[0];
        }

        public virtual void Init(Vector3Int inputShape)
        {
            InputShape = inputShape;
            OutputShape = inputShape;
        }

        public virtual void Release()
        {
        }
    }
}