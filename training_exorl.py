from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from research.dmc import make as make_env
from research.exorl.agent.td3_mtm import TD3Agent
from research.exorl.logger import Logger
from research.exorl.replay_buffer import make_replay_loader
from research.exorl.utils import Every, Timer, Until, set_seed_everywhere
from research.exorl.video import VideoRecorder


def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder):
    """Evaluate the agent and log metrics."""
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = Until(num_eval_episodes)
    while eval_until_episode(episode):
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))
        while not time_step.last():
            with torch.no_grad():
                action = agent.act(time_step.observation, global_step, eval_mode=True)
            time_step = env.step(action)
            video_recorder.record(env)
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f"{global_step}.mp4")

    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("episode_reward", total_reward / episode)
        log("episode_length", step / episode)
        log("step", global_step)


@hydra.main(config_path="configs", config_name="exorl_config.yaml", version_base="1.1")
def main(cfg):
    """Main training loop for ExORL tasks."""
    # Workspace setup
    work_dir = Path.cwd()
    print(f"Workspace: {work_dir}")

    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # Environment setup
    env = make_env(cfg.task, seed=cfg.seed)
    obs_shape = env.observation_spec().shape
    action_shape = env.action_spec().shape

    # Initialize agent
    agent = TD3Agent(
        name=cfg.agent.name,
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        lr=cfg.agent.lr,
        hidden_dim=cfg.agent.hidden_dim,
        critic_target_tau=cfg.agent.critic_target_tau,
        stddev_schedule=cfg.agent.stddev_schedule,
        nstep=cfg.agent.nstep,
        batch_size=cfg.batch_size,
        stddev_clip=cfg.agent.stddev_clip,
        use_tb=cfg.agent.use_tb,
        path=cfg.agent.model_path,
        end_to_end=cfg.agent.end_to_end,
        keep_obs=cfg.agent.keep_obs,
        use_state_action_rep=cfg.agent.use_state_action_rep,
    )

    # Logger setup
    logger = Logger(
        work_dir,
        use_tb=cfg.use_tb,
        config=OmegaConf.to_container(cfg),
        name=cfg.agent.name,
    )

    print("cfg = ", cfg)
    # Replay buffer setup
    print("cfg.replay_buffer_dir = ", cfg.replay_buffer_dir)
    print("cfg.task = ", cfg.task)
    replay_dir = (
        Path(cfg.replay_buffer_dir).resolve() / cfg.task / cfg.expl_agent / "buffer"
    )
    print(f"Replay dir: {replay_dir}")

    replay_loader = make_replay_loader(
        env,
        replay_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
    )
    replay_iter = iter(replay_loader)

    # Video recorder setup
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    # Training loop setup
    timer = Timer()
    global_step = 0

    train_until_step = Until(cfg.num_grad_steps)
    eval_every_step = Every(cfg.eval_every_steps)
    log_every_step = Every(cfg.log_every_steps)

    while train_until_step(global_step):
        if eval_every_step(global_step):
            logger.log("eval_total_time", timer.total_time(), global_step)
            eval(global_step, agent, env, logger, cfg.num_eval_episodes, video_recorder)

        metrics = agent.update(replay_iter, global_step)
        logger.log_metrics(metrics, global_step, ty="train")

        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed_time)
                log("total_time", total_time)
                log("step", global_step)

        global_step += 1

    # Save final model
    model_save_path = work_dir / "final_model.pth"
    torch.save(agent.actor.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")


if __name__ == "__main__":
    main()
