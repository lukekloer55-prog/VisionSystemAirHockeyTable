"""Dashboard service handoff guards, using mocked systemd commands."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import operations_panel as op

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root=Path(self.tmp.name);self.run=self.root/'example';self.run.mkdir()
        (self.run/'mjx_run.json').write_text(json.dumps({'mode':'league'}))
        (self.run/'latest.zip').touch();(self.run/'STOP').touch()
        self.c=SimpleNamespace(root=self.root,code=self.root,lock=threading.RLock(),path=lambda n:self.run,running=lambda p:False)
    def tearDown(self):self.tmp.cleanup()
    @patch.object(op.sys,'platform','linux')
    def test_start(self):
        with patch.object(op,'command',side_effect=['inactive']*3+['Started']) as call:
            self.assertEqual(op.action(self.c,'start',{'name':'example'})['message'],'Started')
            self.assertFalse((self.run/'STOP').exists())
            self.assertIn('--run-dir',call.call_args.args[0])
    @patch.object(op.sys,'platform','linux')
    def test_busy_preserves_stop(self):
        self.c.running=lambda p:True
        with self.assertRaisesRegex(ValueError,'Stop & save'):op.action(self.c,'start',{'name':'example'})
        self.assertTrue((self.run/'STOP').exists())
    @patch.object(op.sys,'platform','linux')
    def test_other_service_blocked(self):
        with patch.object(op,'command',return_value='activating'):
            with self.assertRaisesRegex(ValueError,'service is active'):op.action(self.c,'start',{'name':'example'})
    @patch.object(op.sys,'platform','linux')
    def test_wrong_run_not_stopped(self):
        (self.run/'service.json').write_text('{"unit":"airhockey-league.service"}')
        with patch.object(op,'command',return_value='/other/run'):
            with self.assertRaisesRegex(ValueError,'another run'):op.action(self.c,'stop',{'name':'example'})

if __name__=='__main__':unittest.main()
