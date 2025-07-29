import pygame
import sys

# Initialize pygame
pygame.init()
# screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Keyboard Control Example")
clock = pygame.time.Clock()

running = True
while running:
    # screen.fill((0, 0, 0))  # black background

    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check which keys are pressed
    keys = pygame.key.get_pressed()
    print(keys)
    if keys[pygame.K_w]:
        pass
    if keys[pygame.K_a]:
        print("A pressed")
    if keys[pygame.K_s]:
        print("S pressed")
    if keys[pygame.K_d]:
        print("D pressed")

    # pygame.display.flip()
    clock.tick(60)  # 60 FPS

pygame.quit()
sys.exit()