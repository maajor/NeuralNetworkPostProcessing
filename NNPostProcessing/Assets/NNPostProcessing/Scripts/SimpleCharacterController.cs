using System;
using UnityEngine;

    [RequireComponent(typeof (Rigidbody))]
    [RequireComponent(typeof (CapsuleCollider))]
    public class SimpleCharacterController : MonoBehaviour
    {
        public float Speed = 8.0f;
        public Camera cam;
    
        private Rigidbody m_RigidBody;
        private float m_YRotation;

        private Quaternion m_CharacterTargetRot;
        private Quaternion m_CameraTargetRot;

        private void Start()
        {
            m_RigidBody = GetComponent<Rigidbody>();

            m_CharacterTargetRot = transform.localRotation;
            m_CameraTargetRot = cam.transform.localRotation;
    }


        private void Update()
        {
            if (Mathf.Abs(Time.timeScale) < float.Epsilon) return;
            
            float oldYRotation = transform.eulerAngles.y;

            float yRot = Input.GetAxis("Mouse X") * 2;
            float xRot = Input.GetAxis("Mouse Y") * 2;

            m_CharacterTargetRot *= Quaternion.Euler(0f, yRot, 0f);
            m_CameraTargetRot *= Quaternion.Euler(-xRot, 0f, 0f);

            transform.localRotation = m_CharacterTargetRot;
            cam.transform.localRotation = m_CameraTargetRot;

            Quaternion velRotation = Quaternion.AngleAxis(transform.eulerAngles.y - oldYRotation, Vector3.up);
            m_RigidBody.velocity = velRotation * m_RigidBody.velocity;
    }


        private void FixedUpdate()
        {
            Vector2 input = new Vector2
            {
                x = Input.GetAxis("Horizontal"),
                y = Input.GetAxis("Vertical")
            };

            if ((Mathf.Abs(input.x) > float.Epsilon || Mathf.Abs(input.y) > float.Epsilon))
            {
                Vector3 desiredMove = cam.transform.forward*input.y + cam.transform.right*input.x;
                desiredMove = Vector3.ProjectOnPlane(desiredMove, Vector3.up).normalized;

                desiredMove *= Speed;
                if (m_RigidBody.velocity.sqrMagnitude <
                    (Speed * Speed))
                {
                    m_RigidBody.AddForce(desiredMove, ForceMode.Impulse);
                }
            }
            m_RigidBody.drag = 5f;
        }
    }