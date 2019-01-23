// neural network post-processing
//#define DEBUG_LAYER

using System;
using System.Collections;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.Rendering;


namespace NNPP
{
    public class NNModel
    {
        public List<NNLayerBase> Layers;
        private InputLayer Input;
        private OutputLayer Output;
#if DEBUG_LAYER
        public int debug_layer = 0;
#endif
        public void Load(string name)
        {
            TextAsset text = Resources.Load<TextAsset>("Model/" + name);
            NNModelSerialize modelSerialize = JsonUtility.FromJson<NNModelSerialize>(text.text);

            Layers = new List<NNLayerBase>();
            for (int i = 0; i < modelSerialize.LayerJson.Count; i++)
            {
                var nnlayer = JsonUtility.FromJson(modelSerialize.LayerJson[i],
                    Type.GetType(modelSerialize.LayerTypes[i])) as NNLayerBase;
                nnlayer.FromCache();
                Layers.Add(nnlayer);
                if (i == 0) Input = nnlayer as InputLayer;
            }
            Output = new OutputLayer();
        }

        public void Init(int height, int width)
        {
            Input.Init(new Vector3Int(height, width, Input.InputChannels));
#if !DEBUG_LAYER
            for (int i = 1; i < Layers.Count; i++)
#else
            for (int i = 1; i < debug_layer + 1; i++)
#endif
            {
                if (Layers[i] is Concatenate)
                {
                    Vector3Int input1 = Layers[i - 1].OutputShape;
                    Vector3Int input2 = Layers[(Layers[i] as Concatenate).AlternativeInputId].OutputShape;
                    Layers[i].Init(new Vector3Int(input1.x, input1.y, input1.z + input2.z));
                }
                else
                {
                    Layers[i].Init(Layers[i - 1].OutputShape);
                }
            }
#if !DEBUG_LAYER
            Output.Init(Layers[Layers.Count - 1].OutputShape);
#else
            Output.Init(Layers[debug_layer].OutputShape);
#endif
        }
        private int _height, _width;

        private void Setup(RenderTexture src)
        {
            if (_height != src.height || _width != src.width)
            {
                Init(src.height, src.width);
                _height = src.height;
                _width = src.width;
            }
            Input.src = src;
        }

        public RenderTexture Predict(RenderTexture src)
        {
            Setup(src);
            Input.Run(null);
#if !DEBUG_LAYER
            for (int i = 1; i < Layers.Count; i++)
#else
            for (int i = 1; i < debug_layer + 1; i++)
#endif
            {
                if (Layers[i] is Concatenate)
                {
                    Layers[i].Run(new object[2]
                    {
                        Layers[i - 1].Output,
                        Layers[(Layers[i] as Concatenate).AlternativeInputId].Output,
                    });
                }
                if (Layers[i] is Add)
                {
                    Layers[i].Run(new object[2]
                    {
                        Layers[i - 1].Output,
                        Layers[(Layers[i] as Add).AlternativeInputId].Output,
                    });
                }
                else
                {
                    Layers[i].Run(new object[1] {Layers[i - 1].Output});
                }
            }
#if !DEBUG_LAYER
            Output.Run(new object[1] { Layers[Layers.Count - 1].Output });
#else
            Output.Run(new object[1] { Layers[debug_layer].Output });
#endif
            return Output.outputTex;
        }

        public void Release()
        {
            foreach (var layer in Layers)
            {
                layer.Release();
            }
            Input.Release();
            Output.Release();
        }
    }
}