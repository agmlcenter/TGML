class Logger:
    def __init__(self, log_full_path):
        f = open(log_full_path, 'w')
        print('Saving logs to {}'.format(log_full_path))
        self.file_handle = f

    def log_and_write(self, msg):
        print(msg)
        self.file_handle.write(str(msg) + '\n')
        self.file_handle.flush()