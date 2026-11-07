"""Offline integration tests against the installed Codex CLI and a local model stub."""
import contextlib
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid

import cswitch


class Model(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests.append(body)
        text = 'OFFLINE_MODEL_OK'
        item = {'type': 'message', 'id': 'msg_' + uuid.uuid4().hex, 'role': 'assistant',
                'content': [{'type': 'output_text', 'text': text, 'annotations': []}], 'status': 'completed'}
        response = {'id': 'resp_' + uuid.uuid4().hex, 'object': 'response', 'model': body['model'],
                    'status': 'completed', 'output': [item],
                    'usage': {'input_tokens': 100, 'output_tokens': 5, 'total_tokens': 105}}
        events = [
            {'type': 'response.created', 'response': dict(response, status='in_progress', output=[])},
            {'type': 'response.output_item.added', 'output_index': 0, 'item': dict(item, content=[])},
            {'type': 'response.output_text.delta', 'output_index': 0, 'content_index': 0, 'item_id': item['id'], 'delta': text},
            {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
            {'type': 'response.completed', 'response': response},
        ]
        payload = ''.join('event: ' + e['type'] + '\ndata: ' + json.dumps(e) + '\n\n' for e in events).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class RPC:
    def __init__(self, home, cwd):
        self.log = tempfile.TemporaryFile(mode='w+')
        self.p = subprocess.Popen([cswitch.codex_binary(), 'app-server', '--stdio'],
                                  env=dict(os.environ, CODEX_HOME=str(home)), cwd=cwd,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
        self.q = queue.Queue()
        self.next_id = 0
        self.notifications = []
        def read():
            for line in self.p.stdout:
                try:
                    self.q.put(json.loads(line))
                except ValueError:
                    pass
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        self.call('initialize', {'clientInfo': {'name': 'cswitch_test', 'version': '1'},
                                 'capabilities': {'experimentalApi': True}})
        self.send({'method': 'initialized'})

    def send(self, message):
        self.p.stdin.write(json.dumps(message) + '\n')
        self.p.stdin.flush()

    def get(self):
        try:
            return self.q.get(timeout=30)
        except queue.Empty:
            self.log.seek(0)
            raise AssertionError('RPC timed out: ' + self.log.read()[-3000:])

    def call(self, method, params):
        self.next_id += 1
        n = self.next_id
        self.send({'id': n, 'method': method, 'params': params})
        while True:
            message = self.get()
            if message.get('id') == n:
                if 'error' in message:
                    raise AssertionError(method + ': ' + str(message['error']))
                return message['result']
            self.notifications.append(message)

    def turn(self, tid, text):
        self.call('turn/start', {'threadId': tid, 'input': [{'type': 'text', 'text': text}]})
        while True:
            message = self.get()
            if message.get('method') == 'turn/completed':
                turn = message['params']['turn']
                if turn['status'] != 'completed':
                    raise AssertionError(str(turn))
                return turn['id']

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.p.stdin.close()
        self.p.stdout.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Model)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cswitch-tests-')
        self.root = Path(self.temp.name)
        self.old_home = os.environ.get('CSWITCH_HOME')
        os.environ['CSWITCH_HOME'] = str(self.root/'switch')
        self.work = self.root/'work'
        self.work.mkdir()
        self.homes = {}
        for name in 'abc':
            home = self.root/name
            home.mkdir()
            home.joinpath('config.toml').write_text(f'''model = "test-{name}"
model_provider = "test-{name}"
cli_auth_credentials_store = "file"

[model_providers.test-{name}]
name = "Offline test {name}"
base_url = "http://127.0.0.1:{self.server.server_port}/v1"
wire_api = "responses"
supports_websockets = false

[features]
apps = false
''')
            self.homes[name] = home
            cswitch.ensure_database(home)
        self.settings = self.root/'settings.json'
        cswitch.write_json(self.settings, {'homes': {k: str(v) for k, v in self.homes.items()}})
        self.s = cswitch.Switcher(self.settings)

    def tearDown(self):
        if self.old_home is None:
            os.environ.pop('CSWITCH_HOME', None)
        else:
            os.environ['CSWITCH_HOME'] = self.old_home
        self.temp.cleanup()

    def create(self, mode='paginated', marker='MEMORY_AT_A_4276'):
        with RPC(self.homes['a'], self.work) as rpc:
            result = rpc.call('thread/start', {'cwd': str(self.work), 'historyMode': mode})
            tid = result['thread']['id']
            rpc.turn(tid, 'Remember this project marker: ' + marker)
        return tid

    def resume(self, name, tid, text):
        with RPC(self.homes[name], self.work) as rpc:
            result = rpc.call('thread/resume', {'threadId': tid, 'model': 'test-'+name, 'modelProvider': 'test-'+name})
            self.assertEqual(result['thread']['id'], tid)
            rpc.turn(tid, text)

    def test_paginated_round_trip_and_export_import(self):
        tid = self.create()
        self.s.transfer('b', tid, no_launch=True)
        self.resume('b', tid, 'Remember second marker: MEMORY_AT_B_8391')
        body = json.dumps(Model.requests[-1])
        self.assertIn('MEMORY_AT_A_4276', body)
        self.assertEqual(Model.requests[-1]['model'], 'test-b')
        self.s.transfer('c', tid, no_launch=True)
        self.resume('c', tid, 'Continue project work.')
        body = json.dumps(Model.requests[-1])
        self.assertIn('MEMORY_AT_A_4276', body)
        self.assertIn('MEMORY_AT_B_8391', body)
        self.s.transfer('a', tid, no_launch=True)
        self.resume('a', tid, 'Back at the original environment.')
        self.assertIn('MEMORY_AT_B_8391', json.dumps(Model.requests[-1]))
        package = self.root/'export'
        cswitch.main(['--settings', str(self.settings), 'export', tid, str(package), '--from', 'a'])
        m = cswitch.validate_package(package)
        self.assertEqual(m['thread_id'], tid)
        self.assertFalse(any(p.name in ('config.toml','auth.json') for p in package.rglob('*')))
        cswitch.main(['--settings', str(self.settings), 'import', 'c', str(package), '--replace'])
        self.assertEqual(cswitch.digest(self.homes['a'], cswitch.thread_row(self.homes['a'], tid)),
                         cswitch.digest(self.homes['c'], cswitch.thread_row(self.homes['c'], tid)))

    def test_legacy_round_trip(self):
        tid = self.create('legacy')
        self.s.transfer('c', tid, no_launch=True)
        self.resume('c', tid, 'New marker MEMORY_AT_C_9822')
        self.s.transfer('a', tid, no_launch=True)
        self.resume('a', tid, 'Continue.')
        self.assertIn('MEMORY_AT_A_4276', json.dumps(Model.requests[-1]))
        self.assertIn('MEMORY_AT_C_9822', json.dumps(Model.requests[-1]))

    def test_active_writer_refused(self):
        for mode in ('legacy', 'paginated'):
            with self.subTest(mode=mode):
                tid = self.create(mode)
                with RPC(self.homes['a'], self.work) as rpc:
                    rpc.call('thread/resume', {'threadId': tid})
                    with self.assertRaisesRegex(cswitch.Error, '正在 .* 中使用'):
                        self.s.transfer('b', tid, no_launch=True)
                self.assertIsNone(cswitch.thread_row(self.homes['b'], tid))

    def test_divergent_work_refused_then_backed_up(self):
        tid = self.create()
        self.s.transfer('b', tid, no_launch=True)
        self.resume('a', tid, 'DIVERGENT_A_MARKER')
        self.resume('b', tid, 'DIVERGENT_B_MARKER')
        with self.assertRaisesRegex(cswitch.Error, '不同进度'):
            self.s.transfer('c', tid, no_launch=True)
        with self.assertRaisesRegex(cswitch.Error, '独立修改'):
            self.s.transfer('b', tid, source='a', no_launch=True)
        before = cswitch.digest(self.homes['b'], cswitch.thread_row(self.homes['b'], tid))
        self.s.transfer('b', tid, source='a', replace=True, no_launch=True)
        manifests = list((self.s.root/'backups'/tid).glob('*/previous/manifest.json'))
        self.assertTrue(any(json.loads(p.read_text()).get('fingerprint') == before for p in manifests))

    def test_paginated_fork_and_revert_lineage(self):
        parent = self.create()
        with RPC(self.homes['a'], self.work) as rpc:
            result = rpc.call('thread/fork', {'threadId': parent})
            tid = result['thread']['id']
            rpc.turn(tid, 'FORK_RETAIN_MARKER_2233')
            last = rpc.turn(tid, 'REVERT_REMOVE_MARKER_5577')
            rpc.call('thread/revert', {'threadId': tid, 'beforeTurnId': last})
            original_page = rpc.call('thread/turns/list', {'threadId': tid, 'limit': 100})
        files, _ = cswitch.lineage(self.homes['a'], cswitch.thread_row(self.homes['a'], tid))
        self.assertGreaterEqual(len(files), 2)
        self.s.transfer('b', tid, no_launch=True)
        self.resume('b', tid, 'Check retained context.')
        body = json.dumps(Model.requests[-1])
        self.assertIn('MEMORY_AT_A_4276', body)
        self.assertIn('FORK_RETAIN_MARKER_2233', body)
        self.assertNotIn('REVERT_REMOVE_MARKER_5577', body)
        self.assertIsNone(cswitch.thread_row(self.homes['b'], parent))
        with RPC(self.homes['b'], self.work) as rpc:
            copied_page = rpc.call('thread/turns/list', {'threadId': tid, 'limit': 100})
        # B added one turn; all inherited/reverted source turns are still visible.
        source_ids = {turn['id'] for turn in original_page['data']}
        copied_ids = {turn['id'] for turn in copied_page['data']}
        self.assertTrue(source_ids <= copied_ids)

    def test_compressed_rollout(self):
        tid = self.create()
        p = Path(cswitch.thread_row(self.homes['a'], tid)['rollout_path'])
        subprocess.run(['zstd', '-q', str(p), '-o', str(p)+'.zst'], check=True)
        p.unlink()
        self.s.transfer('b', tid, no_launch=True)
        self.resume('b', tid, 'Read compressed history.')
        self.assertIn('MEMORY_AT_A_4276', json.dumps(Model.requests[-1]))

    def test_full_turn_history_and_credentials_stay_separate(self):
        tid = self.create()
        self.resume('a', tid, 'SECOND_TURN_MARKER')
        self.resume('a', tid, 'THIRD_TURN_MARKER')
        with RPC(self.homes['a'], self.work) as rpc:
            original = rpc.call('thread/read', {'threadId': tid, 'includeTurns': True})
        before = {name: (home/'config.toml').read_bytes() for name, home in self.homes.items()}
        (self.homes['a']/'auth.json').write_text('{"test_secret":"SOURCE_ONLY"}')
        self.s.transfer('c', tid, no_launch=True)
        with RPC(self.homes['c'], self.work) as rpc:
            copied = rpc.call('thread/read', {'threadId': tid, 'includeTurns': True})
        self.assertEqual(len(original['thread']['turns']), 3)
        self.assertEqual(original['thread']['turns'], copied['thread']['turns'])
        for name, home in self.homes.items():
            self.assertEqual(before[name], (home/'config.toml').read_bytes())
        self.assertFalse((self.homes['c']/'auth.json').exists())

    def test_failed_transfer_restores_previous_session(self):
        tid = self.create()
        self.s.transfer('b', tid, no_launch=True)
        before = cswitch.digest(self.homes['b'], cswitch.thread_row(self.homes['b'], tid))
        self.resume('a', tid, 'NEW_SOURCE_PROGRESS')
        actual = cswitch.apply_package
        def fail_after_publish(package, home, restore=False):
            result = actual(package, home, restore)
            if not restore:
                raise cswitch.Error('injected failure after publication')
            return result
        with mock.patch.object(cswitch, 'apply_package', side_effect=fail_after_publish):
            with self.assertRaisesRegex(cswitch.Error, 'injected failure'):
                self.s.transfer('b', tid, no_launch=True)
        self.assertEqual(before, cswitch.digest(self.homes['b'], cswitch.thread_row(self.homes['b'], tid)))
        self.resume('b', tid, 'After failed switch.')
        self.assertNotIn('NEW_SOURCE_PROGRESS', json.dumps(Model.requests[-1]))

    def test_missing_history_dependency_refused(self):
        parent = self.create()
        with RPC(self.homes['a'], self.work) as rpc:
            tid = rpc.call('thread/fork', {'threadId': parent})['thread']['id']
            rpc.turn(tid, 'Fork marker')
        files, _ = cswitch.lineage(self.homes['a'], cswitch.thread_row(self.homes['a'], tid))
        self.assertGreater(len(files), 1)
        files[-1][0].unlink()
        with self.assertRaisesRegex(cswitch.Error, '前序记录'):
            self.s.transfer('b', tid, no_launch=True)
        self.assertIsNone(cswitch.thread_row(self.homes['b'], tid))


if __name__ == '__main__':
    unittest.main(verbosity=2)
