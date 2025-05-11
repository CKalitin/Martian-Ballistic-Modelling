import sys
import pygame

def main():
    image_file = 'Historic Lander Data/Perserverance Charts/perserverance-flight-path-angle.jpg'

    # Initialize Pygame and minimal display
    pygame.init()
    pygame.display.set_mode((1, 1))

    try:
        image = pygame.image.load(image_file).convert_alpha()
    except pygame.error as e:
        print(f"Failed to load image '{image_file}': {e}")
        return

    orig_width, orig_height = image.get_size()
    screen = pygame.display.set_mode((orig_width, orig_height), pygame.RESIZABLE)
    pygame.display.set_caption("Left-click to add points, right-click to remove. Mouse wheel to zoom. ESC/Q to quit.")

    zoom = 1.0
    offset_x, offset_y = 0, 0
    points = []
    clock = pygame.time.Clock()
    running = True

    last_zoom = zoom
    scaled_img = image

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                sx, sy = event.pos
                ix = (sx - offset_x) / zoom
                iy = (sy - offset_y) / zoom
                px, py = int(ix), int(iy)
                if 0 <= px < orig_width and 0 <= py < orig_height:
                    pt = (px, py)
                    if event.button == 1:
                        points.append(pt)
                    else:
                        if points:
                            nearest = min(points, key=lambda p: (p[0]-px)**2 + (p[1]-py)**2)
                            points.remove(nearest)
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                img_x = (mx - offset_x) / zoom
                img_y = (my - offset_y) / zoom
                # Moderate zoom speed, clamp zoom for performance
                factor = 1.2 if event.y > 0 else 1 / 1.2
                zoom = max(0.1, min(zoom * factor, 18))
                offset_x = mx - img_x * zoom
                offset_y = my - img_y * zoom

        # Rescale only on zoom change
        if zoom != last_zoom:
            sw = max(1, int(orig_width * zoom))
            sh = max(1, int(orig_height * zoom))
            scaled_img = pygame.transform.scale(image, (sw, sh))
            last_zoom = zoom

        screen.fill((30, 30, 30))
        screen.blit(scaled_img, (offset_x, offset_y))

        # Draw X markers
        for p in points:
            cx = (p[0] + 0.5) * zoom + offset_x
            cy = (p[1] + 0.5) * zoom + offset_y
            sx = int(cx)
            sy = int(cy)
            s = 5
            pygame.draw.line(screen, (255,0,0), (sx-s, sy-s), (sx+s, sy+s), 2)
            pygame.draw.line(screen, (255,0,0), (sx-s, sy+s), (sx+s, sy-s), 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print(f"Image dimensions: {orig_width} x {orig_height}")
    print("Points (x, y):")
    for p in points:
        p = (p[0], orig_height - 1 - p[1]) # For a y range of 655, you get 0-654, so subtract 1
        print(p)
    print(f"Image dimensions: {orig_width} x {orig_height}")

if __name__ == '__main__':
    main()
