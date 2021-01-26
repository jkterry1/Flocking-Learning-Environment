using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Bird : MonoBehaviour{
    public int id;
    public List<State> states;
    public GameObject birdModel;
    public GameObject leftWing;
    public GameObject rightWing;

    public Bird(int myID){
        id = myID;
        states = new List<State>();
    }

    public void AddModel(GameObject model){
        birdModel = model;
        leftWing = birdModel.transform.Find("body").Find("left wing").gameObject;
        rightWing = birdModel.transform.Find("body").Find("right wing").gameObject;
    }

    public void Step(int curFrame){
        Vector3 leftWingOrientation = new Vector3(states[curFrame].alphaLeft,
                                                  0.0f,
                                                  states[curFrame].betaLeft);
        leftWing.transform.eulerAngles = leftWingOrientation;

        Vector3 rightWingOrientation = new Vector3(-states[curFrame].alphaRight,
                                                   0.0f,
                                                   -states[curFrame].betaRight);
        rightWing.transform.eulerAngles = rightWingOrientation;

        Vector3 pos = new Vector3(states[curFrame].x,
                                  states[curFrame].y,
                                  states[curFrame].z);
        birdModel.transform.position = pos;

        Vector3 orient = new Vector3(states[curFrame].pitch,
                                states[curFrame].yaw,
                                states[curFrame].roll);
        birdModel.transform.eulerAngles = orient;
    }

    public void AddState(State state){
        states.Add(state);
    }
}
