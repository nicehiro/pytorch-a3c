import retro


env = retro.make(game="Airstriker-Genesis")


s = env.reset()
while True:
    env.render()
    a = env.action_space.sample()
    s_, _, d, _ = env.step(a)
    s = s_
    if d:
        s = env.reset()
