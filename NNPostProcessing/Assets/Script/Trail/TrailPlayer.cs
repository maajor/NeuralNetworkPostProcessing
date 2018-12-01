using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrailPlayer : MonoBehaviour {

    public CameraTrail tr;

    public int FrameInterval = 5;
    // Use this for initialization
    void Start () {
		
	}
	
	// Update is called once per frame
	void Update ()
	{
	    int frame = Time.frameCount * FrameInterval;

        if (frame < tr.Positions.Count)
	    {
	        Vector3 pos = tr.Positions[frame];
	        Quaternion rot = tr.Rotations[frame];
            transform.SetPositionAndRotation(pos, rot);
        }
	}
}
