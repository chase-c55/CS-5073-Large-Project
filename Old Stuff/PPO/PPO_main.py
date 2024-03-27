if __name__ == "__main__":
    from stable_baselines3 import PPO
    from typing import Callable
    from PPO_graph_color_env import GraphColoring
    import time

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
          current learning rate depending on remaining progress
        """

        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    env = GraphColoring()
    state, info = env.reset()

    env.render()

    model = PPO("MlpPolicy", env, verbose=1, batch_size=128)
    model.learning_rate = linear_schedule(3e-4)
    model._setup_lr_schedule()

    from stable_baselines3.common.logger import configure

    tmp_path = "./"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv"])
    model.set_logger(new_logger)

    TIME_STEPS = 200_000

    init_time = time.time()
    model.learn(total_timesteps=TIME_STEPS, progress_bar=False)
    env.render()
    print(f"Time taken for {TIME_STEPS} time steps:", time.time() - init_time)
    for i in range(16):
        print("Trial:", i)
        obs, info = env.reset()
        print(env.permute())
        print(env.render())
        model.learn(total_timesteps=50_000, progress_bar=True)


    # import csv
    # csv_file = open("steps.csv", "w")
    # writer = csv.writer(csv_file)
    # writer.writerow(["node_order", "steps"])

    # node_order = [i for i in range(8)]
    # print(node_order)
    
    done = False
    # model.load("ppo_model_200k_16_50k_v2_used_for_new_1k_histo")

    # try:
    #     for i in range(1000):
    #         steps = 0
    #         reward_sum = 0
    #         while not done:
    #             # env.render()
    #             # plt.show()
    #             action, _ = model.predict(state)

    #             state, reward, done, _, info = env.step(action)
    #             # print(state, action, "Reward: ", reward)
    #             # print(env.node_colors)
    #             steps += 1
    #             reward_sum += reward

    #         print("Trial:", i, "Steps:", steps, "Reward:", reward_sum, "Num colors:", len(set(env.node_colors)))
    #         # writer.writerow([node_order, steps])
    #         env.render()
    #         done = False
    #         state, info = env.reset()
    #         node_order = env.permute()
    # except Exception as e:
    #     print("Error happened:", e)
    # finally:
    #     csv_file.close()
    # env.render()

    # model.save(f"ppo_model_{TIME_STEPS}k_steps")
