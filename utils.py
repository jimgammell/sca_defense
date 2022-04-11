log_file = None

def set_log_file(dest):
    global log_file
    log_file = open(dest, 'w')

def log_print(*args, **kwargs):
    global log_file
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)

def print_dict(d, prefix=''):
    for k in d:
        log_print(prefix, '%s:'%(k), sep='', end='')
        if type(d[k]) == dict:
            log_print()
            print_dict(d[k], prefix=prefix+'\t')
        else:
            log_print('', d[k])