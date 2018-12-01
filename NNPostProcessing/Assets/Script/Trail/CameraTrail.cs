using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraTrail : ScriptableObject
{

    public List<Vector3> Positions;
    public List<Quaternion> Rotations;

    public CameraTrail()
    {
        Positions = new List<Vector3>();
        Rotations = new List<Quaternion>();
    }
}
