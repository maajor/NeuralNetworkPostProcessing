// neural network post-processing

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
        protected int KernelId;
        public NNLayerBase()
        {
        }

        /*public virtual void LoadWeight(KerasLayerWeightJson[] weights)
        {

        }*/

        public virtual void Run(object[] input)
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

        public virtual void FromCache()
        {

        }
    }
}