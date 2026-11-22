# Common Kittra workflows.  `make help` lists the targets.
PY      ?= python
DEVICE  ?= cuda
EPISODES ?= 100

.PHONY: help status resume-ppo install install-train test test-fast view expert bench ros-build sim demo \
        data train-act train-diffusion train-matrix train-ppo train-bc eval-experts eval-place eval-il eval-ablation results clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

status: ## what has finished so far and what to run next
	$(PY) scripts/status.py

install: ## Phase 1 dependencies + editable install
	$(PY) -m pip install -r requirements-sim.txt && $(PY) -m pip install -e .

install-train: ## Phase 2 dependencies (torch, lerobot, stable-baselines3)
	$(PY) -m pip install -r requirements-train.txt && $(PY) -m pip install -e .

test: ## all tests (includes slow expert tests)
	$(PY) -m pytest -v

test-fast: ## tests without the slow expert rollouts
	$(PY) -m pytest -m "not slow" -v

# ----------------------------------------------------------------- Phase 1
view: ## open the scene in the MuJoCo viewer
	$(PY) scripts/view_scene.py

expert: ## scripted expert, 50 episodes, nominal + randomized (Phase 1 results)
	$(PY) scripts/run_expert.py --episodes 50 --config nominal
	$(PY) scripts/run_expert.py --episodes 50 --config randomized

bench: ## simulation speed on this machine
	$(PY) scripts/benchmark_sim.py

ros-build: ## build the ROS 2 workspace (source /opt/ros/jazzy/setup.bash and the venv first)
	cd ros2_ws && $(PY) -m colcon build --symlink-install

sim: ## simulated cell in RViz (idle)
	ros2 launch kittra_bringup sim.launch.py

demo: ## simulated cell with the scripted expert running
	ros2 launch kittra_bringup sim.launch.py autopilot:=expert

# ----------------------------------------------------------------- Phase 2
data: ## record all LeRobot datasets (nominal/randomized x 200, randomized 50/100)
	$(PY) scripts/record_lerobot_dataset.py --config nominal    --episodes 200 --overwrite
	$(PY) scripts/record_lerobot_dataset.py --config randomized --episodes 200 --overwrite
	$(PY) scripts/record_lerobot_dataset.py --config randomized --episodes 50  --overwrite
	$(PY) scripts/record_lerobot_dataset.py --config randomized --episodes 100 --overwrite

train-act: ## ACT on the 200-demo randomized dataset
	$(PY) scripts/train_lerobot.py --config configs/train/act.yaml --dataset kittra_randomized --device $(DEVICE)

train-diffusion: ## Diffusion Policy on the 200-demo randomized dataset
	$(PY) scripts/train_lerobot.py --config configs/train/diffusion.yaml --dataset kittra_randomized --device $(DEVICE)

train-matrix: ## full matrix {ACT, Diffusion} x {nominal, randomized 50/100/200}
	for ds in kittra_nominal kittra_randomized kittra_randomized_50 kittra_randomized_100; do \
	  for p in act diffusion; do $(PY) scripts/train_lerobot.py --config configs/train/$$p.yaml --dataset $$ds --device $(DEVICE); done; done

train-ppo: ## PPO placement skill (CPU), checkpoints every 100k steps
	$(PY) scripts/train_ppo.py --config configs/train/ppo_place.yaml

resume-ppo: ## continue an interrupted PPO run
	$(PY) scripts/train_ppo.py --config configs/train/ppo_place.yaml --resume

train-bc: ## behaviour-cloning placement baseline
	$(PY) scripts/train_bc_place.py --config configs/train/bc_place.yaml

eval-experts: ## privileged upper bounds for both tasks
	$(PY) scripts/evaluate.py --policy-type expert --name expert --episodes $(EPISODES)
	$(PY) scripts/evaluate.py --policy-type place_expert --name place_expert --episodes $(EPISODES)

eval-place: ## PPO vs BC on the placement task
	$(PY) scripts/evaluate.py --policy-type sb3 --policy outputs/ppo_place/best_model.zip --name ppo_place --episodes $(EPISODES)
	$(PY) scripts/evaluate.py --policy-type bc --policy outputs/bc_place/bc_place.pt --name bc_place --episodes $(EPISODES)

eval-il: ## every trained ACT/Diffusion checkpoint under nominal/randomized/ood
	for p in act diffusion; do for d in nominal randomized randomized_50 randomized_100; do \
	  ck=outputs/$${p}_$$d/checkpoints/last/pretrained_model; [ -d $$ck ] || continue; \
	  data=$${d%%_*}; n=$$(echo $$d | grep -oE '[0-9]+$$' || echo 200); \
	  $(PY) scripts/evaluate.py --policy-type $$p --policy $$ck --name $${p}_$${data}_$$n \
	    --train-data $$data --n-demos $$n --episodes $(EPISODES) --device $(DEVICE); done; done

eval-ablation: ## single-perturbation ablation for the nominal-trained ACT policy
	$(PY) scripts/evaluate.py --policy-type act --policy outputs/act_nominal/checkpoints/last/pretrained_model \
	  --name act_nominal_200 --train-data nominal --n-demos 200 --conditions ablation --episodes $(EPISODES) --device $(DEVICE)

results: ## build results/phase2_table.md and plots
	$(PY) scripts/make_results_table.py --runs results/runs --out results/phase2_table.md --plots results/plots

clean: ## remove build artefacts (keeps data/outputs/results)
	rm -rf ros2_ws/build ros2_ws/install ros2_ws/log .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
