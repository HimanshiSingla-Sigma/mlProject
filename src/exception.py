import sys

# this is a function for custom exception
# whereever I need to raise exception , i will just call this function
def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    # error_detail.exc_info() function returns three arguments , we are 
    # not interested in the first two 
    # exc_tb -> contains information like -> in which file the error has accured and on which line number
    file_name = exc_tb.tb_frame.f_code.co_filename #file name in which the error has occured
    error_message = "error has occured in script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
    def __str__(self):
        return self.error_message
