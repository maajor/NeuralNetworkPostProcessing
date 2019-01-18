using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NNModelSerialize
{
    public List<string> LayerTypes;
    public List<string> LayerJson;

    public NNModelSerialize()
    {
        LayerTypes = new List<string>();
        LayerJson = new List<string>();
    }
}
