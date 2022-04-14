using UnityEngine;
[RequireComponent(typeof(Camera))]
[ExecuteInEditMode]
public class MirrorFlipCamera : MonoBehaviour
{
    new Camera camera;
    void Awake () {
        camera = GetComponent<Camera>();
    }

    void OnPreCull() {
        if (Application.platform == RuntimePlatform.LinuxPlayer) {
            camera.ResetWorldToCameraMatrix();
            camera.ResetProjectionMatrix();
            Vector3 scale = new Vector3(-1, -1, 1);
            camera.projectionMatrix = camera.projectionMatrix * Matrix4x4.Scale(scale);
        }
    }

    void OnPreRender () {
        if (Application.platform != RuntimePlatform.LinuxPlayer) return;
        GL.invertCulling = false;
    }

    void OnPostRender () {
        if (Application.platform != RuntimePlatform.LinuxPlayer) return;
        GL.invertCulling = true;
    }
}
