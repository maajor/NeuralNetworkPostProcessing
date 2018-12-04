using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    [System.Serializable]
    public class NNLayerBase
    {
        public string Name;
        public int4 InputShape;
        public int4 OutputShape;
        public int4 WeightShape;
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

        public virtual void Init(int4 inputShape)
        {
            InputShape = inputShape;
            OutputShape = inputShape;
        }

        public virtual void Release()
        {
        }
    }
}