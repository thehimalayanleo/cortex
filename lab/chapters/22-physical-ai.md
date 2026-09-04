---
title: "Lab 22: Physical AI: teleoperate, record, and train a policy for a desk robot arm"
kind: permanent
topics: [lab, robotics]
chapter: 22
station: none
recipe: recipes/arm_bridge.py
reading_time: 55 min
---

## What you will be able to do

- Explain what a servo actually does when you send it a goal position, what it reports back, and why a robot arm is a control loop before it is anything to do with learning.
- Compute where the gripper of a six-joint arm is from its joint angles (forward kinematics), check it against the Physical tab's readout, and say why the inverse problem has zero, one, or many answers.
- Record an episode in the LeRobot layout (frames of observations and actions at a fixed rate, plus the task text), and explain what each field is for when a policy is trained on it.
- Describe the three policy families that dominate desk-arm learning today (ACT, diffusion policy, vision-language-action models such as GR00T and π0) by what they predict, what they condition on, and what breaks first.
- Turn an instruction into a motion with a language model as a planner, and say precisely what such a planner can and cannot know about the world.

## The idea in one paragraph

A desk arm is six motors on a serial bus. Each motor holds a position controller: you write a goal in ticks, it reads its own encoder a thousand times a second and pushes toward the goal; you can read the encoder back. Everything above that is software: a forward-kinematics function that turns six angles into a point in space so a person can see what a pose means, a recorder that writes joint angles at a fixed rate so a dataset exists, and a policy, which is any function from recent observations (joint angles, camera frames, a task string) to the next few actions (joint goals). Learning enters at the policy: imitation from recorded episodes is the workhorse, and the reason it works on a desk is that the action space is small (six numbers) and the observations are cheap (one or two webcams). The Physical tab is this stack with the policy left open: the bridge talks to the motors or emulates them, the view is forward kinematics, the recorder writes LeRobot-style frames, and the planner is a language model proposing waypoints, which is the weakest possible policy and a useful baseline for the ones you will train.

## The math

### A servo is a proportional controller with a speed cap

An STS3215 servo holds a 12-bit encoder ($4096$ ticks per turn, $0.088^\circ$ per tick) and a goal register. Its firmware runs

$$u_t = \operatorname{clip}\!\big(K_p\,(g - \theta_t),\ -u_{\max},\ u_{\max}\big),$$

where $g$ is the goal, $\theta_t$ the present position, $u_t$ the drive. With no load the position obeys a first-order lag, $\theta_{t+1} = \theta_t + k\,(g - \theta_t)$ per control tick, which is why the simulator in `arm_bridge.py --sim` is exactly that recursion with a time constant of about 120 ms. Two things follow. First, a step change in the goal produces a smooth motion by itself, so a policy that emits goals at 10 Hz still moves the arm continuously; this is the reason 10 to 30 Hz action rates are enough on a desk. Second, the servo saturates at $u_{\max}$, so a goal far from the present position is followed at constant speed and the lag model is wrong there: the bridge's "goto" eases between poses with a cosine profile so the goal never runs far ahead of the arm, which keeps the motion inside the regime where the lag model holds and keeps the currents low.

Calibration is the affine map from ticks to degrees per joint, $\theta = s\,(n - n_0)\,\tfrac{360}{4096}$ with sign $s$ and offset $n_0$ measured once by holding the arm at a known pose. Without it two arms built from the same kit disagree by an arbitrary offset per joint, and a dataset recorded on one cannot drive the other. This is the single most common reason a downloaded policy "does nothing".

### Forward kinematics of a planar arm on a turntable

The SO-ARM101 is a pan joint about the vertical axis followed by three pitch joints whose axes are parallel (shoulder lift, elbow, wrist flex), then a wrist roll and a gripper. Ignore the roll for position (it does not move the tip). With link lengths $\ell_1, \ell_2, \ell_3$ (upper arm, forearm, wrist plus gripper) and the pitch angles $\alpha_1, \alpha_2, \alpha_3$ measured from vertical, the tip in the arm's plane is

$$r = \ell_1 \sin\alpha_1 + \ell_2 \sin(\alpha_1+\alpha_2) + \ell_3 \sin(\alpha_1+\alpha_2+\alpha_3),\qquad z = h + \ell_1 \cos\alpha_1 + \ell_2 \cos(\alpha_1+\alpha_2) + \ell_3 \cos(\alpha_1+\alpha_2+\alpha_3),$$

with $h$ the base height, and the pan $\phi$ rotates the plane: $x = r\cos\phi$, $y = r\sin\phi$. That is the whole of `fk()` in the server and the reason the view can draw the arm from six numbers. Cumulative angles appear because each joint rotates everything after it; this is the product of rotations written out for the case where all the axes are parallel, and it is what a general kinematic chain computes with $4\times 4$ homogeneous matrices when they are not.

The inverse problem, angles from a target $(r, z)$ and tip orientation, is a two-link problem once you fix the wrist angle: with $d^2 = r'^2 + z'^2$ for the wrist point,

$$\cos\alpha_2 = \frac{d^2 - \ell_1^2 - \ell_2^2}{2\ell_1\ell_2},$$

which has two solutions (elbow up and elbow down) when $|\cos\alpha_2| < 1$, one at the boundary of reach, and none beyond it. Reach is $\ell_1+\ell_2+\ell_3 \approx 0.41$ m for this arm, drawn as the dashed circle in the view. Learned policies never solve this equation; they output joint goals directly, and a planner that outputs joint angles (as the tab's does) sidesteps it too. That is a real limitation: nothing in a joint-space planner knows where the table is.

### An episode is a sequence of (observation, action) pairs at a fixed rate

LeRobot stores a dataset as episodes, each a table with one row per frame at `fps`: `observation.state` (joint angles), `action` (the joint goals that were commanded, usually from a leader arm or the sliders), optional `observation.images.*` (one tensor per camera), a `timestamp`, and per-episode metadata including the task string. The Physical tab's recorder writes the same content as JSON lines (joints, target, tip, time) plus a meta file; converting to the parquet-and-video layout LeRobot's trainer reads is a formatting step, not a modelling one.

The action is the *goal* that was sent, not the position that was reached. This distinction is what makes imitation learning well posed: the policy is trained to reproduce what the operator commanded, and the servo's lag is left to the servo. Recording the reached position as the action instead teaches a policy to lag behind itself.

### Three policies

Behaviour cloning fits $\pi_\theta(a_t \mid o_t)$ by maximum likelihood on the recorded pairs. Its failure is compounding error: a small deviation puts the arm in a state the dataset never covered, the next action is worse, and the error grows roughly linearly in the horizon. The three families in use on desk arms are three answers to that.

*ACT* (Action Chunking with Transformers) predicts a chunk of the next $k$ actions (typically $k = 50$ to $100$ at 50 Hz) from the current observation with a transformer decoder, trained as a conditional VAE so multimodal demonstrations do not average into a mean that is neither. At run time the chunks overlap and are averaged with exponential weights (temporal ensembling). Predicting a chunk cuts the number of decisions per episode by $k$, which is the mechanism against compounding error.

*Diffusion policy* represents $\pi(a_{t:t+k} \mid o_t)$ as a denoising process over the action chunk: start from Gaussian noise and apply the learned denoiser for a few steps. It handles multimodality natively (the samples fall on the modes) at the cost of several forward passes per decision, which is why it runs at 10 Hz where ACT runs at 50.

*Vision-language-action models* (GR00T N1.5, π0, OpenVLA) put a pretrained vision-language backbone in front of an action head (a diffusion or flow-matching head in GR00T and π0), so the task string and the camera frames share a representation that was trained on far more than your episodes. They are what you fine-tune when you have 50 episodes and want generalisation across object positions; the price is a 2 to 3 billion parameter model that needs a real GPU next to the arm, or a remote one on a fast link, and a data-conversion step into the model's expected modality config.

What they all share: the input is a short window of observations, the output is a chunk of joint goals, and the training signal is imitation. Reinforcement learning on real hardware is rare on desk arms because resetting the scene is the expensive step; it appears as a fine-tuning stage on top of imitation (HIL-SERL) rather than from scratch.

### A language model as a planner

The tab's planner sends the model the joint limits, the link lengths, the named poses, and the current angles, and asks for waypoints in joint space. This is a policy with no observation of the world beyond what the prompt says. It can produce a wave, a reach, an open-close, and sequences of poses it was told about, and it will produce a confident grasp of a block whose position it was never given. The right way to read its output is as a *program over poses*, checked by forward kinematics before execution (the tab shows the tip of every waypoint for that reason) and by the joint limits (the server clamps). The step from this to a real VLA is to give the model the camera, which is what the models in the previous section are.

## Build it small

Start the simulator from the Physical tab (or from chat: "connect the arm in sim"). Move the sliders and watch the tip readout: set lift to $45^\circ$ and everything else to $0$, and check $z = h + (\ell_1+\ell_2+\ell_3)\cos 45^\circ$ against the number in the corner. Then:

1. Type a task, press Record, run `reach` then `rest`, stop. Open `~/Cortex/arm/episodes/<id>/frames.jsonl` and find the frame where `target` jumps to the reach pose while `joints` is still near home: that gap is the lag the servo model predicts.
2. Play the episode at speed 2. The trail in the view is the tip path; it should retrace the recording, and it will overshoot slightly on the corners because playback sends the same goals faster than the lag can follow.
3. Ask the planner for "wave hello, then reach forward and open the gripper" and read the waypoints before running them. Then ask for "pick up the cup on the left" and notice that the plan is a plausible motion toward a place the model chose; there is no cup.

The bridge is a run: open its log from the status strip to see the 20 Hz tick, and stop it from the runs list like any other.

## Build it real

Plug the arm's USB-C into this Mac or into the GPU box; the port appears in the Connection card (`/dev/tty.usbmodem…` on macOS, `/dev/ttyACM0` on Linux). Calibrate once: hold the arm straight up with the gripper closed, read the raw ticks from the bridge log, and write `~/Cortex/arm/calibration.json` as `{joint: {offset: ticks_at_zero, sign: ±1}}`. Connect with the port selected; `pyserial` is enough for the STS3215 protocol, and if `lerobot` is installed in the run's environment the bridge uses its bus instead. With a leader arm, a second bridge in read-only mode is the teleoperation source (the next milestone of this tab); with only the follower, the sliders and saved poses are the operator.

For a policy: record 30 to 50 episodes of one task with one webcam, convert to LeRobot's layout, and fine-tune ACT on the 5090 (a few hours) or GR00T N1.5 with its post-training script (longer, and it wants the camera at the resolution it was trained on). Deploy with the policy on the box and the bridge on the Mac, or both on the box; the bridge's `cmd.json` interface is what a policy writes to. For the Nebius track the same recipe runs on a Nebius GPU node with the bridge on the machine the arm is plugged into, and the planner points at Nemotron on Token Factory by setting `NEBIUS_API_KEY`.

## How it goes wrong

- The arm jerks or hums at a pose: the goal is outside the mechanical range even though it is inside the software limits, so the servo pushes against a stop. Tighten the limits for that build.
- A downloaded policy does nothing or sweeps to a wall: calibration offsets differ from the dataset's. Check by driving to the dataset's "home" pose and comparing joints.
- Recorded episodes look right but the trained policy stalls near the object: the demonstrations were too clean (one path each time), so the policy never saw recovery. Record with deliberate small mistakes and corrections.
- The policy works on the training table and fails on another: lighting and background shift. Randomise them while recording, or use a VLA whose backbone has seen more scenes.
- Latency: a policy over a network link adds tens of milliseconds per decision; with action chunks that is fine, with a one-action-per-step policy it is not.

## Measure it

- Tracking error: the mean absolute difference between `target` and `joints` in an episode, in degrees. Rises with speed; should be under $2^\circ$ at rest.
- Success rate over $n \ge 20$ trials with a Wilson interval, per task and per object position; a single number without the interval is not a result.
- Time to completion and number of regrasps, which separate "works" from "works well".
- Recording throughput: frames per second actually written, against `fps`; a recorder that drops frames produces a dataset with a hidden time warp.

## Exercises

1. Derive the elbow-up and elbow-down solutions for a wrist point at $r = 0.2$ m, $z = 0.25$ m with $\ell_1 = 0.1165$, $\ell_2 = 0.135$, $h = 0.0585$, and drive the arm to each with the sliders.
2. Modify the simulator's time constant to 400 ms and record the same motion; measure how the tracking error changes.
3. Write the conversion from `frames.jsonl` to a LeRobot `LeRobotDataset` (one episode, no cameras) and load it with the library.
4. Add a second, read-only bridge that mirrors a leader arm into the follower's `cmd.json`; measure the round-trip latency.
5. Replace the planner's prompt with one that includes a list of object positions and ask for the same grasp; check every waypoint's tip against the object with the forward kinematics.

## Test yourself

1. What does a servo do with a goal position, and why does a 10 Hz action stream still produce smooth motion?
2. Why does the recorder store the commanded goal as the action rather than the reached position?
3. Give the forward-kinematics formula for the tip height of the three pitch joints, and say why the angles are cumulative.
4. How many inverse-kinematics solutions can a two-link planar arm have for a reachable point, and what are they called?
5. What is compounding error in behaviour cloning, and what does action chunking do about it?
6. Why is ACT trained as a conditional VAE?
7. What is temporal ensembling, and what does it trade?
8. Why does a diffusion policy run slower than ACT at inference?
9. What does a VLA add over ACT, and what does it cost?
10. What does the language-model planner in the tab not observe, and what check does the tab apply to its output before running it?

<details><summary>Answers</summary>

1. It runs a proportional controller toward the goal with a speed cap, reading its encoder about a thousand times a second; between goals it keeps converging, so a goal every 100 ms produces a continuous first-order motion.
2. Because the policy must learn what the operator commanded; the servo supplies the lag at run time. Training on reached positions teaches the policy to lag behind itself.
3. $z = h + \ell_1\cos\alpha_1 + \ell_2\cos(\alpha_1+\alpha_2) + \ell_3\cos(\alpha_1+\alpha_2+\alpha_3)$; each joint rotates every link after it, so the orientation of link $i$ is the sum of the joint angles up to $i$.
4. Two (elbow up, elbow down) strictly inside the reach, one at the boundary, none beyond.
5. A small deviation puts the arm in a state outside the data, the next prediction is worse, and the error grows with the horizon. Chunking makes $k$ actions from one decision, so there are $k$ times fewer decisions in which to drift.
6. Demonstrations of the same task are multimodal; a plain regression averages the modes into an action that is neither. The latent variable lets the decoder represent distinct modes.
7. Averaging the overlapping predictions for the same time step from several chunks with exponential weights; it smooths the motion at the cost of reacting more slowly to a change in the scene.
8. It needs several denoising forward passes per decision; ACT decodes once.
9. A pretrained vision-language backbone that ties the task text and the camera to a representation trained on far more data, hence generalisation across scenes and objects; the cost is a 2 to 3 billion parameter model, a real GPU near the arm, and a conversion of your data into its modality config.
10. The world: it sees only limits, link lengths, poses, and the current angles, never the camera. The server clamps joints to the limits and the tab shows each waypoint's tip from forward kinematics so the person can check it.

</details>

## What will change, what will not

The particular policies will change: ACT and diffusion policy are 2023 designs, GR00T and π0 are 2025 ones, and the next generation will be larger, will take more cameras, and will be trained across many arms. The stack under them will not: motors with position controllers and encoders, a bus, a calibration, a fixed-rate recorder of observations and commanded actions, and a policy that maps a short window of observations to a chunk of goals. Compounding error, multimodal demonstrations, calibration mismatch and scene shift are properties of the problem, not of a model, and every new policy is an answer to one of them. The one thing that is genuinely moving is where the policy runs: on-arm inference from a small model, versus a large model on a remote GPU with action chunks hiding the latency, and the answer to that decides how the Physical tab is wired next.

## Read next

- Zhao, Kumar, Levine, Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT, 2023).
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (2023).
- NVIDIA, "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots" (2025), and the N1.5 post-training guide.
- Physical Intelligence, "π0: A Vision-Language-Action Flow Model for General Robot Control" (2024).
- The LeRobot documentation: the dataset format, the SO-ARM101 assembly and calibration pages.
- Luo et al., "SERL / HIL-SERL: sample-efficient RL on real robots" (2024), for what RL adds on top of imitation.
