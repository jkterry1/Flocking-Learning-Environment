using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Vortex : MonoBehaviour
{
    public float time;
    public float x;
    public float y;
    public float z;
    public float roll;
    public float pitch;
    public float yaw;
    public float gamma;
    public GameObject prefab;
    GameObject v;

    public Vortex(float time, float x, float y, float z, float theta, float phi, float psi, float gamma){
      this.x = x; this.y=y; this.z=z;
      this.roll=roll; this.pitch=this.pitch; this.yaw=yaw;
      this.gamma=gamma;
      this.time= time;
    }

    public Vortex(GameObject prefab, float time, float x, float y, float z, float gamma){
      this.x = x; this.y=y; this.z=z;
      this.gamma=gamma;
      this.time= time;
      this.prefab = prefab;
    }

    public void spawn(){
      v = Instantiate(prefab);
      v.transform.position = new Vector3(x, y, z);
      Vector3 orientation = new Vector3(90.0f + roll, pitch, yaw);
      v.transform.eulerAngles = orientation;
    }

    public void remove(){
      Destroy(v);
    }


}
