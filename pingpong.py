import sys
import cv2
import numpy as np
import pygame
import mediapipe as mp
import time
import math
#-----------------------

"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon= int(0.5), trackCon= int(0.5)):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                 self.mpDraw.draw_landmarks(img, handLms,
                                            self.mpHands.HAND_CONNECTIONS)

        return img



    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Move these lines inside the if statement
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox



    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers.count(1)  # return the count of extended fingers


    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



#-------------------------
# Initialize Pygame
pygame.init()

# Initialise Score and font
score1 = 0
score2 = 0
font = pygame.font.Font(None, 36)

# define camera resolution and frame reduction
wCam, hCam = 1024, 576
frameR = 100 # Frame Reduction

# Set up some constants
WIDTH, HEIGHT = 1024, 576
BALL_RADIUS = 10
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 80
FPS = 120

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Create a clock
clock = pygame.time.Clock()

# Define the ball and the paddles
ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_RADIUS * 2, BALL_RADIUS * 2)
paddle1 = pygame.Rect(0, HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
paddle2 = pygame.Rect(WIDTH - PADDLE_WIDTH, HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

# Define the ball speed and the paddle speed
ball_speed = 5
paddle_speed = 2

# Define the direction of the ball (in terms of dx and dy)
dx, dy = ball_speed, ball_speed

# Initialize hand detector
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
detector = handDetector(maxHands=2)

# Initialise score counter
score_surface1 = font.render(str(score1), True, (255, 0, 0))
score_surface2 = font.render(str(score2), True, (0, 0, 255))

# Initialise sound effects
hitsound = pygame.mixer.Sound(r"C:\Users\User\Desktop\28 Jan\Ping_Pong_Ball_Hit.mp3")
scoresound = pygame.mixer.Sound('3.win.wav')

# Reset Global Scores
def reset_scores():
    global score1
    global score2
    score1 = 0
    score2 = 0
    print('score reset')

# Display the start screen
def show_start_screen(score1, score2):
    # Initialise background image
    background_image = pygame.image.load('pingpongbackground.jpg')
    # Resize the background image to the size of the screen
    background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

    # Draw the background image onto the screen
    screen.blit(background_image, (0, 0))
    title_font = pygame.font.Font(None, 72)
    title_surface = title_font.render("Pling Plong", True, (200, 200, 200))
    screen.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2, HEIGHT // 2 - title_surface.get_height() // 2))

    # Render the score onto a surface.
    score_font = pygame.font.Font(None, 36)
    score_surface = score_font.render(f"Previous Score: {score1} - {score2}", True, (200, 200, 200))
    screen.blit(score_surface, (WIDTH // 2 - score_surface.get_width() // 2, HEIGHT // 2 + title_surface.get_height()))

    # Draw the reset bar
    button_rect = pygame.Rect(40, 50, 180, 50)
    pygame.draw.rect(screen, (255, 0, 0), button_rect,0,10)
    button_text = font.render('Reset Score', True, (255, 255, 255))
    screen.blit(button_text, (60, 60))

    # Draw the start text
    start_font = pygame.font.Font(None, 36)
    start_surface = start_font.render("Press any key to start", True, (200, 200, 200))
    screen.blit(start_surface, (WIDTH // 2 - start_surface.get_width() // 2, HEIGHT // 2 + title_surface.get_height() + score_surface.get_height()))

    # Flip the display
    pygame.display.flip()

    # Wait for the user to press a key
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    reset_scores()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYUP:
                return

# Main game loop
show_start_screen(score1, score2)
while True:
    # Event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                show_start_screen(score1, score2)


    # Read camera feed and find hands
    success, img = cap.read()
    img = detector.findHands(img)
    hands = detector.results.multi_hand_landmarks

    # Move the paddles based on hand positions
    if hands:
        for handNo, hand in enumerate(hands):
            lmList, bbox = detector.findPosition(img, handNo) #get hand position
            numFingers = detector.fingersUp()  # get the number of extended fingers
            _, y = lmList[8][1:] # get vertical movement of hand from position (just y coordinate) (Co-Pilots recomendation)

            # Diffrentiate players based on number of fingers up
            if numFingers >= 4:  # Player 1
                paddle1.centery = np.interp(y, (0, hCam), (0, hCam))
            elif numFingers <= 3:  # Player 2
                paddle2.centery = np.interp(y, (0, hCam), (0, hCam))

    # Move the ball
    ball.x += dx
    ball.y += dy

    # Bounce the ball off the top and the bottom
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        hitsound.play()
        dy *= -1

    # Bounce the ball off the paddles
    if ball.colliderect(paddle1) or ball.colliderect(paddle2):
        hitsound.play()
        dx *= -1

    # If the ball goes off the screen, respawn it and update scores
    if ball.left <= 0:
        ball.center = (WIDTH // 2, HEIGHT // 2)
        dx *= -1  # Make sure the ball moves to the right
        score2 += 1  # Player 2 scores
        scoresound.play()
    elif ball.right >= WIDTH:
        ball.center = (WIDTH // 2, HEIGHT // 2)
        dx *= -1  # Make sure the ball moves to the left
        score1 += 1  # Player 1 scores
        scoresound.play()

    # Initialise background image
    background_image = pygame.image.load('C:\\Users\\User\\Desktop\\28 Jan\\pingpongbackground.jpg')
    # Resize the background image to the size of the screen
    background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

    # Draw the background image onto the screen
    screen.blit(background_image, (0, 0))

    # Render the score onto a surface.
    score_surface1 = font.render(str(score1), True, (255, 0, 0))
    score_surface2 = font.render(str(score2), True, (0, 0, 255))

    # Draw everything
    pygame.draw.rect(screen, (255, 0, 0), paddle1,0,10)
    pygame.draw.rect(screen, (0, 0, 255), paddle2,0,10)
    pygame.draw.ellipse(screen, (20, 200, 20), ball)
    pygame.draw.aaline(screen, (200, 200, 200), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

    # Draw the score surfaces onto the screen.
    screen.blit(score_surface1, (WIDTH // 2 - 50, 10))
    screen.blit(score_surface2, (WIDTH // 2 + 20, 10))


    # Flip the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(FPS)