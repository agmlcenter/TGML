def greedy_approach(themap, n, path, golden):
    i = 0
    j = 0
    check = False
    current_pos = themap[0][0]
    last_pos = themap[0][0]
    if current_pos == 'X':
        print("wrong input, please enter a valid map.")
        exit()
    else:
        path.append((i, j))
        golden += int(current_pos)
    
    while i < n and j < n:
        if i < n - 1 and j < n - 1:
            nextpos1 = themap[i][j+1]
            nextpos2 = themap[i+1][j]
        elif i == n - 1 and j < n-1:
            nextpos1 = themap[i][j+1]
            nextpos2 = 'n'
        elif j == n - 1 and i < n-1:
            nextpos1 = 'n'
            nextpos2 = themap[i+1][j]
        else:
            break

        
        if nextpos1 == 'X' and nextpos2 == 'X':
            print("There is no way out!")
            exit()
        elif nextpos1.isnumeric() and nextpos2.isnumeric():
            check = True

        if check:
            if int(nextpos1) >= int(nextpos2):
                nextpos = nextpos1
                path.append((i, j+1))
                j += 1
            elif (int(nextpos1) < int(nextpos2)):
                nextpos = nextpos2
                path.append((i+1, j))
                i += 1
        if not check:
            if (nextpos1 == '!' and nextpos2 == 'X') or (nextpos1.isnumeric()):
                nextpos = nextpos1
                path.append((i, j+1))
                j += 1
            elif (nextpos2 == '!' and nextpos1 == 'X') or nextpos2.isnumeric():
                nextpos = nextpos2
                path.append((i+1, j))
                i += 1
            elif nextpos1 == '!' and nextpos2 == '!':
                nextpos = nextpos1
                path.append((i, j+1))
                j += 1
        if current_pos == '!' and last_pos != '!':
            if nextpos.isnumeric() and gold >= int(nextpos):
                golden -= int(nextpos)
            elif nextpos == '!':
                current_pos = 0
        else:
            if nextpos.isnumeric():
                golden += int(nextpos)
        last_pos = current_pos
        current_pos = nextpos
        check = False   

    print('path:', '->'.join(map(str,path)), '\n', 'number of coins:', golden, sep = '')


def dp_approach(themap, n, i, j, lastpos, savelist, gold):

    if i == n or j == n or themap[i][j] == 'X':
        return []
    
    last_pos_for_dp = '!' if lastpos == '!' else 0
    if (i, j, gold, last_pos_for_dp) in savelist:
        return savelist[(i, j, gold, last_pos_for_dp)].copy()
    
    newgold = gold
    
    if themap[i][j].isnumeric():
        newgold += int(themap[i][j])
    
    current = themap[i][j]
    if lastpos == '!':
        if themap[i][j] != '!':
            newgold = max(0, gold - int(themap[i][j]))
        current = 0
    
    right = dp_approach(themap, n, i, j+1, current, savelist, newgold)
    
    down = dp_approach(themap, n, i+1, j, current, savelist, newgold)
    
    
    right.extend(down)
    
    if len(right) == 0:
        if i == n - 1 and j == n - 1:
            savelist[(i, j, gold, last_pos_for_dp)] = [(newgold, [(i, j)])]
        else:
            savelist[(i, j, gold, last_pos_for_dp)] = []
        return savelist[(i, j, gold, last_pos_for_dp)].copy()
    
    ans = []
    for path in right:
       c = path[1].copy()
       c.append((i, j))
       ans.append((path[0], c))

    savelist[(i, j, gold, last_pos_for_dp)] = ans


    return savelist[(i, j, gold, last_pos_for_dp)].copy()

def max_gold(themap, n):
    gold = 0
    maxgold = 0
    a = dp_approach(themap, n, 0, 0, 0, {}, gold)
    for i in a:
        maxgold = max(maxgold, i[0])
        tup = i[1]
    path = tup[::-1]
    print('path:', '->'.join(map(str,path)), '\n', 'number of max coins:', maxgold, sep = '')

n = int(input('input number of rows and coumns:'))
mymap = list()
path = list()
gold = 0
for i in range(n):
    mymap.append(input().split(" "))
greedy_approach(mymap, n, path, gold)
max_gold(mymap, n)