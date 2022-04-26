using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class State{
    public float x;
    public float y;
    public float z;
    public float roll;
    public float pitch;
    public float yaw;
    public float alphaLeft;
    public float alphaRight;
    public float betaLeft;
    public float betaRight;

    public State(float x, float y, float z, float roll, float pitch, float yaw, float alphaLeft, float alphaRight, float betaLeft, float betaRight){
        this.x = x;
        this.y = y;
        this.z = z;
        this.roll = roll;
        this.pitch = pitch;
        this.yaw = yaw;
        this.alphaLeft = alphaLeft;
        this.alphaRight = alphaRight;
        this.betaLeft = betaLeft;
        this.betaRight = betaRight;
    }
}