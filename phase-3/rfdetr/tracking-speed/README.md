# Moving-Camera Speed Estimation with RF-DETR + ByteTrack + Ego-Motion Compensation

## Overview

The `rfdetr_tracking_speed_vector_additions` notebook builds directly on the work developed toward the end of the `rfdetr_tracking_speed_improvements` notebook.

Before extending the system to a moving camera, we first developed a **still-camera speed estimator**. That earlier estimator was designed to resemble a live deployment pipeline as closely as possible. It avoided manual calibration, homography, and affine correction, and instead estimated object speed using only information available at inference time:

- **RF-DETR detections**
- **ByteTrack object tracks**
- **predetermined object dimensions**
- **bounding box width and height**
- a carefully chosen **representative point** inside each bounding box

The main idea was to estimate a local **meters-per-pixel scale** from known object dimensions and the size of the detected bounding box. Once that local scale was available, the frame-to-frame motion of the representative point could be converted from pixel displacement into real-world motion. This gave us a practical, class-aware speed estimator that could run without scene-specific setup.

That still-camera estimator became the foundation for everything in this notebook.

---

## From Still-Camera Speed Estimation to Moving-Camera Speed Estimation

With a **still camera**, the motion observed from a tracked vehicle comes mostly from the vehicle itself. In that setting, estimating speed is comparatively straightforward:

1. detect the object
2. track it over time
3. estimate local scale from its bounding box
4. convert tracked image displacement into real-world motion
5. compute speed

However, once the **camera itself begins moving**, the observed motion is no longer just the vehicle’s motion. It becomes a mixture of:

- **vehicle motion**
- **camera ego-motion**

That changes the problem fundamentally.

Instead of estimating only a scalar **speed**, we now need to estimate **2D velocity** so that direction is available as well. Once both the object motion and camera motion are represented as vectors in the same coordinate frame, we can compensate for camera motion using **vector addition**.

So the conceptual progression across the two notebooks is:

- first, learn how to estimate speed when the **camera is still**
- then, preserve that same bbox-dimension-based foundation while extending it to work when the **camera is moving**

This notebook is about that second step.

---

## What Stayed the Same

Even though this notebook introduces moving-camera compensation, it intentionally keeps the same underlying speed-estimation philosophy developed in the previous notebook:

- use **known object dimensions**
- compare them to **pixel dimensions of the bounding box**
- estimate a **local scale**
- use tracked image motion to estimate real-world motion

In other words, we did **not** abandon the still-camera estimator. We built on top of it.

The moving-camera extension adds a new ingredient:

- estimate the **camera’s direction of motion** from the **background**
- combine that direction with a known **camera speed magnitude**
- compensate using **vector addition**

---

## Main Goal

The goal of the `rfdetr_tracking_speed_vector_additions` notebook is to extend the earlier still-camera estimator into a moving-camera estimator that can:

- work on **drone videos**
- preserve the original **bbox-dimension-based speed-estimation foundation**
- require as few additional inputs as possible
- estimate camera direction from the video itself when possible
- compensate for camera motion using **2D velocity vectors**
- produce a clean **annotated output video**
- move toward a form that resembles **deployment-time inference**

---

## High-Level Pipeline

The final pipeline has three main parts:

### 1. Object Detection and Tracking
We use:

- **RF-DETR** for object detection
- **ByteTrack** for multi-object tracking

This gives us:
- bounding boxes
- class labels
- persistent track IDs across frames

### 2. Relative Object Velocity Estimation
For each tracked object, we:

- choose a **representative point** inside the box
- estimate a **local meters-per-pixel scale** from object priors and bbox dimensions
- measure frame-to-frame image displacement
- convert that displacement into a **2D relative velocity vector**

This is still the same basic speed-estimation philosophy used in the earlier still-camera work.

### 3. Camera Ego-Motion Compensation
For moving-camera videos, we:

- estimate the **dominant background motion direction**
- convert that to the **camera motion direction**
- combine that direction with the known **camera speed magnitude**
- form a **camera velocity vector**
- compensate using:

$$
\mathbf{v}_{object,world} = \mathbf{v}_{object,relative} + \mathbf{v}_{camera}
$$

The final displayed speed is:

$$
speed = \|\mathbf{v}_{object,world}\|
$$

So although many details are refined, the central idea is still:

**estimate both motions as vectors, then use vector addition to recover the object’s actual motion.**

---

# Project Evolution in This Notebook

## Phase 1 — Start from a Still-Camera Baseline

We first verify that the relative speed estimator works correctly when the camera is not moving.

This is important because it gives us a clean reference. If the still-camera estimator is unstable, then any moving-camera compensation built on top of it will also be unstable.

### What this baseline does
It:
- runs detection and tracking
- estimates relative object velocity
- computes speed from bbox-based local scale
- checks whether the resulting speeds are stable and plausible

### Why this matters
This becomes the baseline we compare all moving-camera runs against.

In our experiments, the still-camera output behaved well and produced a stable estimate near the expected motion pattern, which meant we were ready to extend the method.

---

## Phase 2 — Move from Speed to Velocity

A scalar speed is not enough once the camera moves.

Why?

Because if the camera is moving, we need to know:
- **how fast**
- and in **which direction**

So the notebook upgrades the estimator from:
- speed magnitude only

to:
- **2D velocity components**
  - $v_x$
  - $v_y$

This lets us reason about motion direction and makes compensation possible.

---

## Phase 3 — Estimate Camera Direction from the Background

Instead of manually telling the system which way the camera is moving, we try to recover that from the video itself.

### How we do it
We:
- mask out detected/tracked objects
- track **background points** with optical flow
- estimate the **dominant background motion direction**
- smooth that direction over time

This gives us the apparent **background flow direction**.

Because background motion is the visual effect of camera motion, we then convert it into the **camera motion direction** by flipping the sign.

### Why this works
If the camera moves left, the background appears to move right.
If the camera moves up, the background appears to move down.

So the background gives us a usable ego-motion direction signal without requiring fixed landmarks.

---

## Phase 4 — Compensate Camera Motion with Vector Addition

Once we have:
- object relative velocity
- camera direction
- known camera speed magnitude

we can form the camera velocity vector and compensate.

This is the key conceptual step of the notebook.

The compensation is:

$$
\mathbf{v}_{object,world} = \mathbf{v}_{object,relative} + \mathbf{v}_{camera}
$$

This allows us to move from:
- “how the object appears to move in a moving camera”
to:
- “how the object is actually moving”

---

# Important Design Decisions

## 1. Keep the Original Speed-Estimation Foundation

We explicitly did **not** switch to:
- homography-based calibration
- map landmarks
- fixed-scene physical references

Instead, we kept the original live-style idea:

- use **predetermined object dimensions**
- compare against **bbox width and height**
- estimate local scale directly from the object box

This makes the method more portable for drone videos captured in different places.

---

## 2. Use Background Motion Instead of Manual Direction Input

The camera speed magnitude is assumed to be known at deployment.

But instead of manually supplying direction, we estimate it from the video itself using background optical flow.

This reduces input requirements and makes the system more robust.

---

## 3. Work in 2D Image-Aligned Ground-Plane Coordinates

The notebook uses a consistent 2D coordinate convention:

- +x = image right
- +y = image down

All motion vectors are expressed in this shared coordinate system before compensation.

This is essential, because vector addition only makes sense if both vectors are in the same frame.

---

# Key Experiments in the Notebook

## Experiment A — Still vs Moving Left vs Moving Right

We first tested three videos:

- camera still
- camera moving left at 5 km/h
- camera moving right at 5 km/h

while the car itself moved at a constant 30 km/h.

### Why this was useful
These controlled tests let us check:
- whether the ego-direction estimate pointed the correct way
- whether compensation moved the estimated speed back toward the still-camera baseline

### What we found
- ego-direction estimation worked very well
- left/right background direction was recovered cleanly
- vector compensation worked
- but the compensation magnitude needed some tuning

This led to the introduction of a compensation scaling term, **alpha**.

---

## Experiment B — Compensation Magnitude Sweep (`alpha`)

We found that raw vector compensation was directionally correct, but a small calibration factor improved results.

So we introduced:

$$
\mathbf{v}_{camera,eff} = \alpha \mathbf{v}_{camera}
$$

### Why alpha was introduced
In practice, the measured object motion does not behave as a perfect direct sum of:
- raw object motion
- raw background flow

Some ego motion is already partially absorbed by:
- box motion
- tracking behavior
- representative-point choice
- local scale estimation

So alpha gives us a lightweight calibration knob.

### Results
- for **left/right** motion, the best alpha was around **0.90**
- for **up/down** motion, the best alpha was higher, around **1.60**

This showed that compensation magnitude was **direction-dependent**.

---

## Experiment C — Moving Up and Moving Down

We then added:

- camera moving down at 5 km/h
- camera moving up at 5 km/h

This tested whether the pipeline generalized beyond horizontal motion.

### What we found
The ego-direction estimation again worked very well:
- moving down → strong vertical ego estimate
- moving up → strong vertical ego estimate

So the background-based ego module generalized successfully.

However, the compensation behavior was not identical to left/right:
- up/down preferred a different alpha
- especially moving up behaved differently than moving down

This suggested that the relative object-motion estimator is not perfectly symmetric across motion directions.

---

# What We Learned

## 1. The moving-camera method works
The notebook successfully extends the still-camera estimator to moving-camera video.

## 2. Ego-direction estimation from the background is reliable
The background optical-flow method consistently recovered the correct motion direction.

## 3. Compensation is still based on vector addition
Even after all refinements, the core logic remains:

- estimate object relative velocity
- estimate camera velocity
- combine them with vector addition

## 4. A single universal compensation gain did not perfectly generalize
Left/right and up/down preferred different compensation magnitudes.

This led to the final practical design:
- use one learned alpha for mostly horizontal motion
- another learned alpha for mostly vertical motion
- blend between them based on camera direction

---

# Final Standalone Pipeline

The final cell in the notebook produces a **self-contained end-to-end speed estimator**.

## Inputs
It takes:
- an input video
- a known camera speed magnitude
- the trained RF-DETR detector
- the class map

## What it does
It runs:

### Pass 1 — Ego estimation
- detect and track objects
- mask them out
- estimate smoothed background motion direction

### Pass 2 — Final estimation and rendering
- detect and track objects again
- estimate relative object velocity
- estimate effective camera-motion vector
- compensate with vector addition
- render a clean output video

## Output video contents
For each object, the final video shows:
- bounding box
- track ID
- class label
- speed estimate

And in the top-left corner, it shows:
- a smoothed camera ego arrow
- or a dot if the camera is still

## What is intentionally not shown
To keep the output clean, the final video does **not** show:
- representative-point dot
- trajectory tail
- $v_x$ / $v_y$ text
- confidence scores

---

# Final Compensation Strategy

The final pipeline uses a **direction-aware alpha blend**.

We learned:
- horizontal motion preferred one alpha
- vertical motion preferred another

So instead of forcing one global value, we compute:

$$
\alpha_{eff} = \alpha_{lr}|d_x| + \alpha_{ud}|d_y|
$$

where:
- $\alpha_{lr}$ is the left/right compensation gain
- $\alpha_{ud}$ is the up/down compensation gain
- $d_x, d_y$ are the camera direction components

This gives:
- mostly horizontal camera motion → alpha close to horizontal setting
- mostly vertical camera motion → alpha close to vertical setting
- diagonal motion → interpolated behavior

---