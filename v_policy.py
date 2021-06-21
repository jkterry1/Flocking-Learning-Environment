N = 5


'''
#policy to stay in a v formation, complicated version
def v_policy(N, agent, obs, wingspan = 0.80):
    id = int(agent[-1])
    a = [0.0, 0.0, 0.0, 0.0, 0.0]
    thrust = basic_flying_policy(obs)

    if id == N//2:
        #print("middle bird")
        return [thrust, 0.0,0.0,0.0,0.0]

    #convert wingspan to normalized obs value
    wingspan = 0.5 + wingspan / 100.0

    #want to be in the highest upward force area
    #find position that it should be in (behind and to the right/left
    #of bird in front of it)
    #1m behind, 2 wingspans to the side

    #find position of bird in front
    #it should be one of the two closest birds, whichever is in front
    x1 = obs[20]
    #x2 = obs[29]
    u = obs[14]
    v = obs[15]
    w = obs[16]
    b_l = obs[11]
    b_r = obs[13]
    #print("bl: ", b_l)
    #print("br: ", b_r)

    #if x1 is in front of the current bird, this is the bird you are "following"
    if x1 > 0:
        x = x1
        y = obs[20]
        z = obs[21]
        #The first half of birds should be to the left
        if id <= N//2:
            #y should be 1m back
            #x should be one wingspan to the left

            #If x too far left
            if x < 0.5 + wingspan:
                #print("x too far left")
                #check if already moving to the right/wing is already up,
                #if it is, don't change anything
                if v <= 0.5 and b_l <= 0.75:
                    #move beta left up
                    a[2] = 0.08
                    #if beta right is up, move it down
                    if b_r > 0.5:
                        a[4] = -0.08

            #If x too far right
            if x > 0.5 + wingspan:
                #print("x too far right")
                #check if already moving to the left/wing is already up,
                #if it is, don't change anything
                if v >= 0.5 and b_r <= 0.5:
                    #move beta left down, beta right up
                    a[2] = -0.08
                    a[4] = 0.08

            a[0] = thrust
            return a
'''




#policy to make sure the birds are staying within some range
def fly_within_range(obs, low = 99.0):
    #basic idea: if the bird is below the low range, require that it go up
    #vertical velocity:

    #find normalized low/high values:
    low = low/500.0

    z = obs[6]
    w = obs[16]
    if z < low:
        return 1.0
    return 0.0;

#policy to keep bird in the air
def basic_flying_policy(obs):
    #basic idea: if the bird is moving down too quickly, add some forward thrust
    #vertical velocity:
    w = obs[16]
    #print(w)
    if w < 0.505:
        return 1.0

    return 0.0
