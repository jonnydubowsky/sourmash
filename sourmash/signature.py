#! /usr/bin/env python
"""
Save and load MinHash sketches in a JSON format, along with some metadata.
"""
from __future__ import print_function
import hashlib
import weakref

import gzip
import bz2file
import io
import sys

from .logging import error


from .logging import error
from .minhash import MinHash

from ._compat import to_bytes
from ._lowlevel import ffi, lib
from .utils import RustObject, rustcall, decode_str


SIGNATURE_VERSION=0.4


sig_refs = weakref.WeakKeyDictionary()
mhs_refs = weakref.WeakKeyDictionary()


class SourmashSignature(RustObject):
    "Main class for signature information."
    _name = ''
    filename = ''

    def __init__(self, minhash, name='', filename=''):
        self._objptr = lib.signature_new()

        if name:
            self._name = name
        if filename:
            self.filename = filename

        self.minhash = minhash

        self.__dealloc_func__ = lib.signature_free

    @property
    def minhash(self):
        return MinHash._from_objptr(self._methodcall(lib.signature_first_mh), shared=True)

    @minhash.setter
    def minhash(self, value):
        # TODO: validate value is a MinHash
        self._methodcall(lib.signature_push_mh, value._objptr)

    def __hash__(self):
        return hash(self.md5sum())

    def __str__(self):
        name = self.name()
        md5pref = self.md5sum()[:8]
        if name != md5pref:
            return "SourmashSignature('{}', {})".format(name, md5pref)
        return "SourmashSignature({})".format(md5pref)
    __repr__ = __str__

    def minhashes(self):
        size = ffi.new("uintptr_t *")
        mhs_ptr = self._methodcall(lib.signature_get_mhs, size)
        size = ffi.unpack(size, 1)[0]

        mhs = []
        for i in range(size):
            mh = MinHash._from_objptr(mhs_ptr[i], shared=True)
            mhs.append(mh)
#            mhs_refs[mh] = mh

        return mhs

    def md5sum(self):
        "Calculate md5 hash of the bottom sketch, specifically."
        m = hashlib.md5()
        m.update(str(self.minhash.ksize).encode('ascii'))
        for k in self.minhash.get_mins():
            m.update(str(k).encode('utf-8'))
        return m.hexdigest()

    def __eq__(self, other):
        return self._methodcall(lib.signature_eq, other._objptr)

    @property
    def _name(self):
        return decode_str(self._methodcall(lib.signature_get_name), free=True)

    @_name.setter
    def _name(self, value):
        self._methodcall(lib.signature_set_name, to_bytes(value))

    def name(self):
        "Return as nice a name as possible, defaulting to md5 prefix."
        name = self._name
        filename = self.filename

        if name:
            return name
        elif filename:
            return filename
        else:
            return self.md5sum()[:8]

    @property
    def filename(self):
        return decode_str(self._methodcall(lib.signature_get_filename), free=True)

    @filename.setter
    def filename(self, value):
        self._methodcall(lib.signature_set_filename, to_bytes(value))

    @property
    def license(self):
        return decode_str(self._methodcall(lib.signature_get_license), free=True)

    def _display_name(self, max_length):
        name = self._name
        filename = self.filename

        if name:
            if len(name) > max_length:
                name = name[:max_length - 3] + '...'
        elif filename:
            name = filename
            if len(name) > max_length:
                name = '...' + name[-max_length + 3:]
        else:
            name = self.md5sum()[:8]
        assert len(name) <= max_length
        return name

    def _save(self):
        "Return metadata and a dictionary containing the sketch info."
        e = dict(self.d)
        minhash = self.minhash

        sketch = {}
        sketch['ksize'] = int(minhash.ksize)
        sketch['num'] = minhash.num
        sketch['max_hash'] = minhash.max_hash
        sketch['seed'] = int(minhash.seed)
        if self.minhash.track_abundance:
            values = minhash.get_mins(with_abundance=True)
            sketch['mins'] = list(map(int, values.keys()))
            sketch['abundances'] = list(map(int, values.values()))
        else:
            sketch['mins'] = list(map(int, minhash.get_mins()))
        sketch['md5sum'] = self.md5sum()

        if minhash.is_protein:
            sketch['molecule'] = 'protein'
        else:
            sketch['molecule'] = 'DNA'

        e['signature'] = sketch

        return self.d.get('name'), self.d.get('filename'), sketch

    def similarity(self, other, ignore_abundance=False, downsample=False):
        "Compute similarity with the other MinHash signature."
        try:
            return self.minhash.similarity(other.minhash, ignore_abundance)
        except ValueError as e:
            if 'mismatch in max_hash' in str(e) and downsample:
                xx = self.minhash.downsample_max_hash(other.minhash)
                yy = other.minhash.downsample_max_hash(self.minhash)
                return xx.similarity(yy, ignore_abundance)
            else:
                raise

    def jaccard(self, other):
        "Compute Jaccard similarity with the other MinHash signature."
        return self.minhash.similarity(other.minhash, True)

    def contained_by(self, other, downsample=False):
        "Compute containment by the other signature. Note: ignores abundance."
        try:
            return self.minhash.contained_by(other.minhash)
        except ValueError as e:
            if 'mismatch in max_hash' in str(e) and downsample:
                xx = self.minhash.downsample_max_hash(other.minhash)
                yy = other.minhash.downsample_max_hash(self.minhash)
                return xx.contained_by(yy)
            else:
                raise


def _guess_open(filename):
    """
    Make a best-effort guess as to how to parse the given sequence file.

    Handles '-' as shortcut for stdin.
    Deals with .gz and .bz2 as well as plain text.
    """
    magic_dict = {
        b"\x1f\x8b\x08": "gz",
        b"\x42\x5a\x68": "bz2",
    }  # Inspired by http://stackoverflow.com/a/13044946/1585509

    if filename == '-':
        filename = '/dev/stdin'

    bufferedfile = io.open(file=filename, mode='rb', buffering=8192)
    num_bytes_to_peek = max(len(x) for x in magic_dict)
    file_start = bufferedfile.peek(num_bytes_to_peek)
    compression = None
    for magic, ftype in magic_dict.items():
        if file_start.startswith(magic):
            compression = ftype
            break
    if compression is 'bz2':
        sigfile = bz2file.BZ2File(filename=bufferedfile)
    elif compression is 'gz':
        if not bufferedfile.seekable():
            bufferedfile.close()
            raise ValueError("gziped data not streamable, pipe through zcat \
                            first")
        sigfile = gzip.GzipFile(filename=filename)
    else:
        sigfile = bufferedfile

    return sigfile


def load_signatures(data, ksize=None, select_moltype=None,
                    ignore_md5sum=False, do_raise=False):
    """Load a JSON string with signatures into classes.

    Returns list of SourmashSignature objects.

    Note, the order is not necessarily the same as what is in the source file.
    """
    if ksize:
        ksize = int(ksize)

    if not data:
        return

    is_fp = False
    if hasattr(data, 'find') and data.find('sourmash_signature') == -1:   # filename
        done = False
        try:                                  # is it a file handle?
            data.read
            is_fp = True
            done = True
        except AttributeError:
            pass

        # not a file handle - treat it like a filename.
        if not done:
            try:
                data = _guess_open(data)
                is_fp = True
                done = True
            except OSError as excinfo:
                error(str(excinfo))
                if do_raise:
                    raise
                return
    else:  # file-like
        if hasattr(data, 'mode'):  # file handler
            if 't' in data.mode:  # need to reopen handler as binary
                if sys.version_info >= (3, ):
                    data = data.buffer

    size = ffi.new("uintptr_t *")

    try:
        data = data.read()
    except AttributeError:
        pass

    try:
        # JSON format
        if is_fp:
            sigs_ptr = rustcall(lib.signatures_load_buffer, data, ignore_md5sum, size)
            #fp_c = ffi.cast("FILE *", data)
            #sigs_ptr = rustcall(lib.signatures_load_file, fp_c, ignore_md5sum, size)
        else:
            if hasattr(data, 'encode'):
                sigs_ptr = rustcall(lib.signatures_load_buffer, data.encode('utf-8'), ignore_md5sum, size)
            else:
                sigs_ptr = rustcall(lib.signatures_load_buffer, data, ignore_md5sum, size)

        size = ffi.unpack(size, 1)[0]

        sigs = []
        for i in range(size):
            sig = SourmashSignature._from_objptr(sigs_ptr[i], shared=True)
            sigs.append(sig)
            sig_refs[sig] = sigs

        for sig in sigs:
            for minhash in sig.minhashes():
                if not ksize or ksize == minhash.ksize:
                    if not select_moltype or \
                         minhash.is_molecule_type(select_moltype):
                        yield sig
                        break
    except Exception as e:
        error("Error in parsing signature; quitting.")
        error("Exception: {}", str(e))
        if do_raise:
            raise
#    finally:
#        if is_fp:
#            data.close()


def load_one_signature(data, ksize=None, select_moltype=None,
                       ignore_md5sum=False):
    sigiter = load_signatures(data, ksize=ksize,
                              select_moltype=select_moltype,
                              ignore_md5sum=ignore_md5sum)

    try:
        first_sig = next(sigiter)
    except StopIteration:
        raise ValueError("no signatures to load")

    try:
        next(sigiter)
    except StopIteration:
        return first_sig

    raise ValueError("expected to load exactly one signature")


def save_signatures(siglist, fp=None):
    "Save multiple signatures into a JSON string (or into file handle 'fp')"
    collected = [obj._get_objptr() for obj in siglist]
    siglist_c = ffi.new("Signature*[]", collected)

    if fp is None:
        buf = rustcall(lib.signatures_save_buffer, siglist_c, len(collected))
    else:
        #fp_c = ffi.cast("FILE *", fp)
        #buf = rustcall(lib.signatures_save_file, siglist_c, len(collected), fp_c)
        buf = rustcall(lib.signatures_save_buffer, siglist_c, len(collected))
        result = decode_str(buf, free=True)
        fp.write(result)
        return None

    return decode_str(buf, free=True)
