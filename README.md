# VIT Bhopal Racer

A small racing simulator whose only missing piece is the driver - and you
do not write the driver, you train it. The circuit is traced onto a
satellite image of the VIT Bhopal campus: down the main road past Vit
Bhopal Square and the Open Auditorium, along the diagonal road below
Academic Block-2, and back up past the football ground and the Boys Hostel.

You design an agent - what it sees, what it is paid for, how it is
trained. So does your friend. So does whichever LLM you want to test.
Everyone's agent learns against identical physics, and `evaluate.py`
decides whose actually got round faster.

## Setup

```bash
pip install -r requirements.txt

python train.py --driver my_driver --workers 8   # run the iterations
python play.py --drivers my_driver               # watch what you trained
python evaluate.py --drivers my_driver baseline  # settle the argument
```

Everything is plain Python plus numpy and pygame. The neural network, the
backpropagation and both learning algorithms are in this repo, in numpy, in
about fifteen hundred readable lines. No torch, no gym, no compilation
step, nothing to configure.

## Files

```
train.py           run the iterations - the one that makes a driver
evaluate.py        headless scoring - the one that settles arguments
play.py            open a window and race
track_editor.py    click a new circuit onto the map image

drivers/
  my_driver.py     >>> YOUR AGENT GOES HERE <<<
  baseline.py      a deliberately slow hand-written opponent to beat
  human.py         arrow-key control

racer/
  agent_api.py     the contract: features in, reward out. Read this first.
  env.py           episodes: reset, step, reward, done
  policy.py        the network, the chain rule, Adam
  learn.py         PPO and evolution strategies - the iterations themselves
  driver_api.py    the older contract, for hand-written drivers
  track.py         waypoints -> smooth circuit, walls, sensors
  car.py           physics: bicycle model + friction circle
  race.py          the simulation loop, laps, collisions, scoring
  render.py        the pygame view
  tournament.py    time trials and championships

runs/
  my_driver/       best.npz, last.npz, log.csv - written by train.py

track_data/
  vit_bhopal.json  the circuit, as waypoints in map-pixel coordinates
  map.png          the satellite image
```

## The interface

A hand-written driver answers one question: *what do I do now?* A learned
driver makes three choices, and those are what you actually compete on:

```python
class Driver(Agent):
    def features(self, obs): ...   # what the network is allowed to see
    def reward(self, step): ...    # what you are paying it to do
    train = {...}                  # how the search is run
```

Everything the agent may know arrives in `obs`, exactly as before (full
list in `racer/driver_api.py`):

| field | meaning |
|---|---|
| `speed` | m/s |
| `lateral` | metres off the centreline, + = left |
| `heading_error` | your heading minus the track's, radians |
| `curvature` | 1/radius here, + = left bend |
| `preview` | `(forward, left, width, curvature)` at 5, 10, 15, 20, 30, 40, 55, 70, 90, 120 m ahead, in your car's frame |
| `rays` | distance to the walls at 11 angles from -90° to +90° |
| `opponents` | other cars, positioned in your frame |
| `slip` | 0 gripping, 1 at the limit, above 1 sliding |
| `on_track` | False means you are in the dirt with half the grip |

`features(obs)` picks from that and returns a numpy vector. The stock one
returns 55 numbers. The network turns them into two: steering, and a
single accelerator axis where positive is throttle and negative is brake.

`reward(step)` is scored once per decision. `step` carries `ds` (metres of
track gained), `dt`, `off_track_time`, `wall_hit`, `collision`, `stalled`,
and both observations. The default pays for distance and fines everything
that makes distance unsustainable.

`prepare(track_info, ...)` runs once before each race and hands you the
entire circuit as numpy arrays - centreline, width, curvature, arc-length.
Precomputing a racing line there costs you nothing at runtime, and feeding
the network its offset from *that* instead of from the centreline is the
single biggest idea in this repo that is not already implemented.

The engine catches exceptions from your agent, so a crash costs you the
race instead of killing everyone else's.

## Training

```bash
python train.py --driver my_driver --iterations 500 --workers 8
```

Each iteration the agent drives a few thousand steps with a little noise on
the controls, gets scored, and has its weights nudged. Every tenth
iteration it is sent out for a real time trial from the grid; if that lap
is the fastest yet, the weights land in `runs/my_driver/best.npz`, which is
the file `play.py` and `evaluate.py` race. Ctrl+C saves on the way out and
`--resume` picks it back up.

```
 iter     steps    reward   metres  crash  noise    sec     time trial
    5     20,520      34.8     75.0   100%  0.392    1.2
   30    123,120     386.9    408.9     0%  0.367    1.0     1:26.250 *
  120    492,480     455.1    494.3     0%  0.284    1.0     1:08.133 *
  600  2,462,400     511.6    552.3    14%  0.147    0.8     1:04.117
```

`metres` is how far it got per episode and is the number that moves first.
`crash` is the fraction of episodes that ended in the scenery - it goes to
zero long before the driver is any good. `noise` is how much randomness is
still on the controls, and it shrinks as the agent grows confident. A `*`
means that lap was a new best and was saved.

Starting a fresh run in a directory that already has a `best.npz` moves it
to `previous_best.npz` rather than overwriting it.

### Watching it learn

```bash
python train.py --driver my_driver --workers 8 --watch
```

`--watch` opens a window next to the training table. You cannot watch every
step - a 600-iteration run is roughly 23 hours of simulated driving crammed
into 13 minutes - so what the window shows is the *newest weights* driving
at normal speed. Each time an iteration writes a checkpoint the viewer
swaps it in underneath the moving car, so one continuous car goes round the
circuit getting visibly better: wandering, then late-braking into the
hairpin, then flowing through it.

### Watching the whole population at once

```bash
python train.py --driver my_driver --arena
```

Evolution does not train one driver, it runs a crowd of thirty-two and keeps
what the good ones did. Normally that crowd is spread across worker
processes and you never see it. `--arena` puts all of them in one race
instead: same starting point, driving simultaneously, passing through each
other like ghosts. They fan out, the hopeless ones spear off into the
scenery and fade to grey where they died, and the survivors carry on. Then
the generation ends and the next thirty-two take the grid.

`--arena` picks `--algo ga` unless you choose otherwise, because that is the
one where the picture is literally true: the cars named `elite00`-`elite07`
in the leaderboard are last generation's best, back again unchanged, racing
against their own children.

The camera follows whoever is furthest ahead. SPACE pauses, `1`-`4`
fast-forward. In the arena each candidate is scored over one episode rather
than three, because otherwise you wait three times as long between
generations; pass `--episodes 3` if you would rather have the cleaner
ranking. `--horizon 300` roughly halves the wait again.

This is the real algorithm, not an animation of one - every car on screen
is a candidate actually being scored. Two things to be honest about:

- **It is much slower than training properly.** Thirty-two cars in one
  process gets nowhere near `--workers 12` headless. Use the arena to see
  what your reward is really rewarding; use headless when you want a fast
  agent.
- **Under `--algo es` and `--algo cem`, "dying" is a picture rather than the
  mechanism.** A crashed car still votes under `es`; it just votes for going
  somewhere else. Only under `--algo ga` do the survivors literally carry
  on into the next generation, which is why the arena picks `ga` by default.

### Watching from another terminal

The live viewer also works against a run already going:

```bash
python play.py --drivers my_driver --live
```

`--live` follows `runs/<driver>/last.npz` (use `--live best` for the record
holder instead), waits politely if training has not written anything yet,
survives a half-written file, and starts the next race when one ends. The
bottom of the window shows which iteration you are looking at.

Four algorithms, all of which just repeat *try, score, adjust*. They differ
in what "adjust" means:

- `--algo ppo` (default) learns from **gradients**. There is one driver. It
  drives with a little randomness on the controls, and the weights are
  nudged towards whichever of its own wobbles paid off better than expected.
  No population at all. Much the fastest to a good lap time - use it first.

- `--algo ga` is a **genetic algorithm**, and it is the one that works the
  way people picture it. Thirty-two separate drivers per generation. The
  best `elite` of them survive into the next generation *unchanged*, as
  themselves. Every other place is filled by a child: two parents are
  chosen by tournament (grab three at random, keep the best), a coin is
  flipped per weight to decide which parent it comes from, and then a few
  weights are nudged at random. That nudge - the mutation - is the only
  source of anything new.

- `--algo es` keeps a population too, but nobody survives. Every candidate
  is a wobble around one shared set of weights; they are all scored, all of
  them vote (weighted by rank), the centre slides, and thirty-two fresh
  wobbles are drawn. Robust, and the best of the three across cores.

- `--algo cem` is `es` with blunter selection: the best `elite` are averaged
  into a single new centre and everyone else is discarded. Still no
  survivors - the average is a new driver that never actually raced.

The distinction that matters: under `ga` the cars on screen are *individuals
with a lineage*. Under `es` and `cem` they are disposable samples, and the
thing being improved is a point they were scattered around.

Under evolution the `reward` and `metres` columns bounce around a lot from
one generation to the next. That is not instability, it is the starting
point: every generation draws a fresh random place on the circuit, and a
generation that starts on the back straight scores far better than one that
starts at the hairpin. Within a generation the comparison is exact - all
thirty-two candidates drive the same start - but across generations the
only comparable number is the `time trial` column, which always runs from
the grid. Watch that one.

Useful flags: `--workers N` (one process per core - this is the difference
between five minutes and an hour), `--traffic baseline baseline` to train
with other cars on track, `--horizon` for episode length, `--steps` for
experience per iteration.

Episodes mostly start at a **random point on the circuit** at a random
survivable speed, because an agent that always starts on the grid spends
its first hundred thousand steps relearning turn one and never sees the
hairpin. One episode in six starts properly from the line so the standing
start is learned too.

## Scoring

```bash
python evaluate.py --drivers my_driver friend_ai rival_ai
```

**Time trial** runs each driver alone. This is the honest measure of pace.

**Championship** runs everyone together over several races, rotating the
grid. This measures racecraft, which is a different skill: a controller
that time-trialled at 34 s a lap can still finish *last* in a grid race,
stuck behind two cars lapping at 105 s, because it has no overtaking logic
at all. An agent trained alone has never seen another car - watch the
`hits` column, which is where the trained agent above leaks its time. Train
with `--traffic` if the championship is the argument you want to win.

Add `--start spread` to space cars around the lap so a quick car is not
trapped in traffic from the first corner.

Hand-written drivers still race. `baseline.py` is one, `human.py` is
another, and a `Driver` subclass with a `drive()` method works exactly as
it always did - the comparison is between drivers, however they were made.

## Where the lap time is

Do these in order. Skipping ahead does not work.

1. **Train the defaults, unchanged.** 300 iterations, no edits. It will be
   scruffy but it will lap. That is your control experiment - every idea
   below has to beat it.
2. **Read the reward.** `w_offtrack` and `w_crash` are the two knobs with
   personality: too soft and it learns to cut every corner, too hard and it
   crawls round terrified of the kerbs. Move one at a time and retrain.
3. **Look at what `features` is not telling it.** The default sees ten
   preview points and eleven rays. It cannot see how far through the corner
   it is, where the apex is, or what the track does after the next bend.
   One good feature beats a hundred more iterations.
4. **Precompute a racing line** in `prepare()` and feed the network its
   offset from that. Outside on entry, clip the apex, unwind on exit. This
   is genuinely hard, and it is where someone who knows the circuit still
   beats a generic answer.
5. **Train with traffic** and use the opponent features. Contact bleeds
   your speed too, so punting people off is not free.

Steps 1–3 get you most of the way. Step 4 is the interesting one.

## Editing the circuit

```bash
python track_editor.py
```

Left-click to add or drag a waypoint, right-click to delete, `[` and `]` to
change the track width at that point, `P` to preview the real driving
surface, `S` to save. Waypoints are stored in map-pixel coordinates, so what
you click is what you get; the conversion to metres happens at load time via
`meters_per_pixel` in the JSON.

To race a different place entirely, drop a new screenshot into `track_data/`,
point the JSON's `image` at it, and trace away. Agents are trained per
circuit - retrain after you move the waypoints.

## Tuning knobs

`CarSpec` in `racer/car.py` holds the whole vehicle: mass, engine force,
brake force, steering lock, and the grip coefficients on and off track.
Lower `grip_on` to make the driving harder; raise `grip_off` to make
mistakes cheaper. All drivers always share one spec, so the comparison
stays fair.
