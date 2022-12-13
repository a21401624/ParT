import logging
import time
import os
import argparse

class Logger:
    def __init__(self, base_dir, name=__name__):
        # 创建一个loggger
        self.__name = name
        self.logger = logging.getLogger(self.__name)
        self.logger.setLevel(logging.DEBUG)
        self.dir=base_dir

        # 创建一个handler，用于写入日志文件
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        logname=os.path.join(self.dir, time.strftime('%Y%m%d_%H%M%S') + ".log")

        fh = logging.FileHandler(logname, mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
        fh.setLevel(logging.DEBUG)

        # 创建一个handler，用于将日志输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s-%(filename)s-[line:%(lineno)d]'
                                      '-%(levelname)s: %(message)s',
                                      datefmt='%d %b %Y %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    @property
    def get_log(self):
        """定义一个函数，回调logger实例"""
        return self.logger


if __name__=='__main__':
    parser = argparse.ArgumentParser("指定日志存储位置")
    parser.add_argument('-b', '--base_dir', default='work_dirs')
    args = parser.parse_args()
    base_dir = args.base_dir
    log = Logger(base_dir, name=__name__).get_log
    log.info("INFO")
    log.error('ERROR')
