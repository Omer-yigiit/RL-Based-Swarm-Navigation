import random

class Agent:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def move(self):
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])

        self.x += dx
        self.y += dy
    
    def get_position(self):
        return (self.x, self.y)

def main():
    agents = [Agent(random.randint(0, 10), random.randint(0, 10)) for _ in range(5)]
    steps = 10
    for step in range(steps):
        print(f"steps:{step}")
        for i, agent in enumerate(agents):
            agent.move()
            print(f"Agent {i} position: {agent.get_position()}")

if __name__ == "__main__":
    main()