class BaseParser:
    application = None

    def __init__(self, application):
        """
        The constructor for the parser.
        """
        self.application = application

    def parse_frame(self, output):
        """
        This function will be called on every frame.
        :param output: the frame
        """
        raise Exception
