import time

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1

    print('时间到了')

if __name__ == '__main__':
    countdown(25 * 60) # 专注时长为25分钟
