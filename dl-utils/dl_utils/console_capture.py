'''
Provide objects that allow capturing stdout/stderr for later replay.
Intended for IPython notebooks, in which if we close the browser window 
with a running kernel all subsequent console output is lost

Idea taken from
http://stackoverflow.com/questions/29119657/ipython-notebook-keep-printing-to-notebook-output-after-closing-browser/29170902#29170902

See also https://github.com/ipython/ipython/issues/4140
and http://nbviewer.jupyter.org/gist/minrk/4563193
'''

from __future__ import print_function
import sys
import re
import tempfile
import os
import threading
from StringIO import StringIO
from tempfile import mkstemp
from time import sleep


if sys.version[0] != '2':
    basestring = str
    xrange = range

_REAL_STDOUT = sys.stdout
_REAL_STDERR = sys.stderr


def clean_string( buf ):
    '''
    Process backspaces and carriage returns in a string, eating up
    the chunks that would be overwritten by them if the string were
    printed to a terminal
    '''
    # Process carriage returns (except \r\n windows-type eol)
    buf = re.sub('[^\r\n]+\r(?!\n)', '', buf)
    # Remove backspaces at the beginning of a line
    buf = re.sub( '^\b+|(?<=[\n])\b+', '', buf )
    # Process backspaces
    while '\b' in buf:
        buf = re.sub('[^\b]\b', '', buf)
    return buf


# -------------------------------------------------------------------------

class OutputDest( object ):
    """
    A file-like object that can print both to stdout and to a file.
    """

    def __init__(self, out=None, console=True):
        self._do_console = console
        self.console = _REAL_STDOUT
        if out is None:
            self.log = StringIO()
        elif isinstance(out,basestring):
            self.log = open(out,'w')
        else:
            self.log = out

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stdout = _REAL_STDOUT
        sys.stderr = _REAL_STDERR
        self.close()

    def write(self, message):
        #idx = message.find( '\m' )
        #if idx:
        #    message = message[idx:]
        #while :
        #    message = re.sub( r'[^\b][\b]', '' ) 
        self.log.write(message)
        self.log.flush()
        if self._do_console:
            self.console.write(message)
        
    def close(self):
        if self.log is not None:
            self.log.close()
            self.log = None

    def truncate(self,*args):
        self.log.truncate(*args)

    def __getattr__(self, attr):
        return getattr(self.console, attr)


# -------------------------------------------------------------------------


class ConsoleCapture( object ):
    '''
    An object that captures the console output and redirects it to a 
    temporal file (and optionally also print it out to console).
    It can be started & stopped at will.
    '''
    def __init__(self):
        self._on = False
        self.fname = None

    def start( self, console=True, name=None, dir=None ):
        """
        Start capturing console output into a file
         :param console: whether to also print all out also to console
         :param name: middle part of the filename (default is "notebook")
         :param dir: directory where to write the logfile 
        """
        logname = '{}-'.format( name or 'notebook' )
        f, self.fname = mkstemp( prefix=logname, suffix='.log', 
                                 dir=dir or os.getcwd(), text=True )
        self.log = OutputDest( os.fdopen(f,'w'), console )
        self._on = True
        self.log.__enter__()

    def stop( self ):
        """
        Stop console capture
        """
        if self._on:
            self.log.__exit__()
            self._on = False
        return self

    def reset( self ):
        """
        Delete all captured data so far. Keep capturing
        """
        if self._on:
            self.logger.truncate(0)
        return self

    def remove( self ):
        """
        Stop capture and remove the captured file
        """
        if self._on:
            self.stop()
        if self.fname:
            os.unlink( self.fname )
            self.fname = None
        return self

    def reprint( self, hdr=None, chunked=None ):
        """
        Print out to the real console the captured data, with an optional header
        """
        if hdr:
            _REAL_STDOUT.write( hdr )
        buf = self.data
        if not chunked:
            clear_output()
            _REAL_STDOUT.write( buf )
            _REAL_STDOUT.flush()
        else:
            for i in range(0,len(buf),256):
                _REAL_STDOUT.write( " CHUNK! ")
                _REAL_STDOUT.write( buf[i:i+256] )
                _REAL_STDOUT.flush()
                sleep( chunked )
        return self

    @property
    def data( self ):
        """
        Return the captured data
        """
        if self.fname:
            with open(self.fname,'r') as f:
                return f.read()

# -------------------------------------------------------------------------

class ConsoleCaptureCtx( ConsoleCapture ):
    '''
    A console capture object to be used as a context manager.
    Two methods can be used inside or outside the "with" context to obtain
    the data:
      * `output()` will print to console the captured output, with an
         optional header
      * `data` (available as a property) will return it
    '''

    def __init__( self, verbose=True, name=None, logdir=None, delete=True,
                  notebook=False ):
        self._lock = threading.RLock()
        self._args = verbose, name, logdir
        self._result = None
        self._delete = delete
        self._notebook = notebook
        super(ConsoleCapture,self).__init__()

    def __enter__( self ):
        with self._lock:
            self.start( *self._args )
            return self

    def __exit__( self, *args ):
        with self._lock:
            self._result = super(ConsoleCaptureCtx,self).data
            if self._delete:
                self.remove()

    @property
    def data( self ):
        '''
        Return the captured data. Override parent so that we can return data 
        even outside the context, when the file does not exist anymore (thanks 
        to our cached copy)
        '''
        with self._lock:
            return self._result if self._result is not None else super(ConsoleCaptureCtx,self).data
        

# -------------------------------------------------------------------------

class ThrStatus( object ):
    '''An enum-like object to hold the thread status'''
    __slots__ = ('CREATED','RUNNING','CLOSING','ENDED','ABORTED','REAPED')
    CREATED,RUNNING,ENDED,ABORTED,REAPED = range(5)
    


class ProcessingThread( threading.Thread ):
    '''
    A thread that wraps a callable and captures its stdout/stderr,
    so that it can be printed out from the main thread.
     1. Create it, passing the callable & args
     2. Start it with the `run()` method
     3. Check if it is running with the `status` attribute
     4. At any moment print out the output generated by the callable by
        calling `output()`. A header will also be printed, indicating the
        thread state.
    When calling `output()` after the thread has finished, it is automatically
    joined.
    '''
    
    def __init__( self, _callable, *args, **kwargs ):
        '''
        Initialize it with the callable to execute plus its arguments
        '''
        super(ProcessingThread,self).__init__()
        self._call = _callable, args, kwargs
        self._st = ThrStatus.CREATED
        self._ctx = None
        self._kwargs = { 'verbose': True,
                         'delete': True }


    def run(self):
        '''
        Thread execution entry point: execute the processing callable, 
        capturing its output
        '''
        with ConsoleCaptureCtx( **self._kwargs ) as ctx:
            self.fname = ctx.fname
            self._ctx = ctx
            self._st = ThrStatus.RUNNING
            try:
                self._call[0]( *self._call[1], **self._call[2] )
            except Exception:
                self._st = ThrStatus.ABORTED
                raise
        self._st = ThrStatus.ENDED


    def set_args( self, **kwargs ):
        '''
        Set execution arguments
        '''
        self._kwargs.update( kwargs )
        return self._kwargs


    def close( self ):
        '''
        If the thread has finished or aborted, collect it
        '''
        if self._st in (ThrStatus.ENDED, ThrStatus.ABORTED):
            self.join()
            self._st = ThrStatus.REAPED

    @property
    def status(self):
        '''
        Return the processing thread running status
        '''
        return self._st

    @property
    def data(self):
        '''
        Return the produced data
        '''
        return self._ctx.data if self._st != ThrStatus.CREATED else None

    @property
    def logname(self):
        '''
        Return the name of the logfile
        '''
        return self._ctx.fname if self._st != ThrStatus.CREATED else None


    def reprint(self, **kwargs):
        '''
        Print out the thread current status & the produced output so far.
        To be called from the main thread.
        When the processing thread has finished, collect it.
        '''
        if not self._ctx:
            return None
        status = ("RUNNING" if self._st == ThrStatus.RUNNING else 
                  "ABORTED" if self._st == ThrStatus.ABORTED else 
                  "DONE" if self._st in (ThrStatus.ENDED,ThrStatus.REAPED) else 
                  "STATUS: {}".format(self._st) )

        self._ctx.reprint( "\r----- {} -----\n".format(status), **kwargs )
        self.close()


# -------------------------------------------------------------------------


class ProcessWrap( object ):
    '''
    A class to wrap a (possibly long running) process and collect all its
    standard output.
    '''
    def __init__( self, _callable, *args, **kwargs ):
        '''
        Create the object by giving it the callable to run and all the
        arguments that must be passed to it
        '''
        self._p = _callable, args, kwargs
        self.tout = None

    def _block( self ):
        while self.tproc.status == ThrStatus.RUNNING:
            sleep( 0.01 )
        self.tproc.close()

    def process( self, verbose=True, delete=True, block=True ):
        '''
        Launch the process
         :param verbose: whether to also output to console
         :param delete: delete the logfile holding the output upon termination
         :param block: block the call while the processing thread is running
        '''
        if verbose:
            print( "Launching process ... " )
            sys.stdout.flush()
        # Create the thread
        self.tproc = ProcessingThread( self._p[0], *self._p[1], **self._p[2] )
        self.tproc.set_args( verbose=verbose, delete=delete )
        # Start the thread
        self.tproc.start()
        # Wait till the thread confirms it has started
        while self.tproc.status == ThrStatus.CREATED:
            sleep( 0.01 )
        # If blocking, wait until the thread finishes
        if block:
            self._block()


    def show( self, block=False, chunked=False, clean=False ):
        '''
        Print the output generated by the processing thread
        '''
        self.tproc.reprint( chunked=0.01 if chunked else None, clean=clean )
        if block:
            self._block()

    @property
    def data( self ):
        '''
        Return the data captured. Will work even if the process has ended.
        '''
        return self.tproc.data

    @property
    def logname( self ):
        '''
        Return the name of the logfile, or `None` if there isn't one
        '''
        return self.tproc.logname

    # -------------------



    def start( self, verbose=True, delete=True, block=True ):        
        self.tproc = ProcessingThread( self._p[0], *self._p[1], **self._p[2] )
        self.tproc.verbose( verbose ).delete( delete ).start()
        while self.tproc.status == ThrStatus.CREATED:
            sleep( 0.01 )

    def show_( self ):
        '''
        Print out the output generated by the processing thread so far
        '''
        self.tproc.reprint()

    def async_show( self, all=False, interval=1.0 ):
        if self.tout:
            self.tout.stop()
        self.tout = OutputThread( self.tproc.fname, all, interval ) 
        self.tout.start()

    def async_stop( self ):
        if self.tout:
            self.tout.stop()



# -------------------------------------------------------------------------

class OutputThread( threading.Thread ):
    '''
    '''
    def __init__( self, fname, all=True, interval=1.0 ):
        super(OutputThread,self).__init__()
        self._fname = fname
        self._st = ThrStatus.CREATED
        self._all = all
        self._interval = interval
        self._lock = threading.Lock()

    def run( self ):
        with self._lock:
            self._st = ThrStatus.RUNNING
        with open( self._fname, 'r' ) as f:
            if not self._all:
                f.seek(0,2)
            while self._on == ThrStatus.RUNNING:
                l = f.readline()
                if l:
                    _REAL_STDOUT.write( l )
                if self._st == ThrStatus.RUNNING:
                    sleep( self._interval )
        with self._lock:
            self._st = ThrStatus.ENDED
        

    def stop( self ):
        '''
        Tell the ouput thread to stop
        '''
        with self._lock:
            self._st = ThrStatus.CLOSING
        self.join()


