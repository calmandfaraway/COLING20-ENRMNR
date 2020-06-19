
def seg_pa(content='', m=20, n=10):     # 以长度m为窗口截取字符串，每步长为n
    # m, n = 20, 10
    # content = 'With the death of the driver who plowed his truck ' \
    #           'through dozens of French civilians in Nice, it may take' \
    #           ' authorities a while to get to fully understand what motivated ' \
    #           'the attack. The broader picture MADRID A Spanish court found ' \
    #           '21 people guilty of involvement in the 2004 Madrid train ' \
    #           'bombings but cleared three men of masterminding Europe deadliest Islamist ' \
    #           'attack Premier League champions Leicester are one of the more ' \
    #           'than 20 clubs celebrating the contribution of refugees to football ' \
    #           'this weekend Much has been made of the ties between ' \
    #           'criminality and radicalisation. But what about the growing intersection between ' \
    #           'drugs and terrorism Brussels is on lockdown after a series ' \
    #           'of terrorist attacks struck the city Tuesday morning: one at ' \
    #           'a metro stop near the headquarters of the European Union ' \
    #           'and two at the airport. At least 31 people are ' \
    #           'dead and dozens more are injured Attorney General Jeff Sessions ' \
    #           'says the Justice Department will crack down on violent gangs' \
    #           'Mounting evidence indicates that Al Qaeda may have been behind ' \
    #           'the March 11th bombings in Madri The start of 2016 ' \
    #           'saw the highest number of terrorism deaths in Western Europe ' \
    #           'since 2004'
    
    start, end = 0, m
    pad_id = 0
    content = content.lower().strip()
    con_list = content.split(' ')
    # print(con_list)
    # print(len(con_list))
    sents = []
    while True:
        segment = []
        if len(con_list) >= m:
            # print(con_list[start:end])
            # print(len(con_list[start:end]))
            sents.append(con_list[start:end])
            con_list = con_list[start + n:]

        else:  # 长度不足m
            # print('else:')
            # margin = m - len(con_list)
            # print(con_list + [pad_id] * margin)
            sents.append(con_list)
            break

    return sents
