"""Regression tests for the GPU layout shown in the README."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_overview as app

ROOT = Path(__file__).resolve().parents[1]


class ParserTests(unittest.TestCase):
    def test_nvidia(self):
        text = (ROOT / 'examples/nvidia-smi.txt').read_text()
        gpus, procs, driver = app.parse_nvidia(text)
        self.assertEqual(driver, '580.178.04')
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[0].bus_id, '00000000:04:00.0')
        self.assertEqual(gpus[0].vram_used_mib, 10559)
        self.assertEqual(gpus[0].vram_total_mib, 16376)
        self.assertEqual(gpus[0].temperature_c, 46)
        self.assertEqual(gpus[0].power_w, 89)
        self.assertEqual(gpus[0].gpu_util_pct, 0)
        self.assertEqual({p.pid for p in procs}, {4556, 11221})
        self.assertEqual(next(p for p in procs if p.pid == 11221).vram_mib, 10534)

    def test_amd(self):
        text = (ROOT / 'examples/amd-smi.txt').read_text()
        gpus, procs, rocm = app.parse_amd(text)
        self.assertEqual(rocm, '10.1.0')
        self.assertEqual(len(gpus), 3)
        self.assertEqual([gpu.index for gpu in gpus], [0, 1, 2])
        self.assertEqual(gpus[0].vram_used_mib, 20068)
        self.assertEqual(gpus[0].vram_total_mib, 32624)
        self.assertEqual(gpus[0].gpu_util_pct, 52)
        self.assertEqual(gpus[1].fan_pct, 31.76)
        self.assertEqual(gpus[1].gpu_util_pct, 73)
        self.assertIsNone(gpus[2].gpu_util_pct)
        self.assertTrue(gpus[2].shared_memory)
        self.assertEqual([p.pid for p in procs], [11221, 11221])
        self.assertEqual([p.vram_mib for p in procs], [18.9 * 1024, 24.4 * 1024])

    def test_aggregate_and_draw(self):
        sample = app.collect(ROOT / 'examples/nvidia-smi.txt', ROOT / 'examples/amd-smi.txt', ROOT / 'examples/top.txt')
        self.assertEqual(len(sample['gpus']), 4)
        self.assertEqual(sample['summary']['discrete_gpu_count'], 3)
        self.assertEqual(sample['summary']['dedicated_vram_used_mib'], 10559 + 20068 + 25828)
        self.assertEqual(sample['summary']['dedicated_vram_total_mib'], 16376 + 32624 * 2)
        self.assertEqual(len([p for p in sample['processes'] if p['pid'] == 11221]), 3)
        self.assertEqual(sample['warnings'], [])
        self.assertEqual(sample['system']['cpu_busy_pct'], 36.0)
        self.assertEqual(sample['system']['ram_used_mib'], 25000)
        self.assertEqual(sample['system']['ram_total_mib'], 65536)
        self.assertIn('CPU non-idle:', app.draw(sample))
        self.assertIn('RAM used:', app.draw(sample))
        self.assertIn('AMD1', app.draw(sample))
        self.assertIn('shared', app.draw(sample))
        # Each discrete GPU memory bar is enclosed in [ ].
        for gpu in (g for g in sample['gpus'] if not g['shared_memory']):
            line = next(row for row in app.draw(sample).splitlines() if row.startswith(gpu['label'] + ' '))
            self.assertRegex(line, r'\[[=.]{12}\]$')

    def test_top_takes_second_sample(self):
        text = (ROOT / 'examples/top.txt').read_text()
        system = app.parse_top(text)
        self.assertEqual(system['cpu_busy_pct'], 36.0)
        self.assertEqual(system['ram_used_mib'], 25000.0)
        self.assertEqual(system['ram_total_mib'], 65536.0)

    def test_top_kib_units(self):
        text = "%Cpu(s): 1.3 us, 3.3 sy, 95.4 id\nKiB Mem : 8388608 total, 6291456 used, 0 free\n"
        system = app.parse_top(text)
        self.assertAlmostEqual(system['cpu_busy_pct'], 4.6)
        self.assertEqual(system['ram_used_mib'], 6144)
        self.assertEqual(system['ram_total_mib'], 8192)

    def test_top_older_percent_notation_and_missing(self):
        text = "Cpu(s): 10.0%us, 5.0%sy, 85.0%id\nGiB Mem : 16.0 total, 4.0 used, 12.0 free\n"
        self.assertEqual(app.parse_top(text)['cpu_busy_pct'], 15.0)
        self.assertEqual(app.parse_top(text)['ram_used_mib'], 4096)
        self.assertEqual(app.parse_top(''), {'cpu_busy_pct': None, 'ram_used_mib': None,
                                             'ram_total_mib': None})

    def test_top_from_local_system(self):
        output, error = app.run_command(['top', '-b', '-n', '2', '-d', '0.2'])
        if error:
            self.skipTest(error)
        system = app.parse_top(output)
        self.assertIsNotNone(system['cpu_busy_pct'])
        self.assertIsNotNone(system['ram_total_mib'])
        self.assertIsNotNone(system['ram_used_mib'])
        self.assertGreater(system['ram_total_mib'], 0)

    def test_no_data(self):
        gpus, procs, driver = app.parse_nvidia('')
        self.assertEqual((gpus, procs, driver), ([], [], None))
        gpus, procs, rocm = app.parse_amd('')
        self.assertEqual((gpus, procs, rocm), ([], [], None))
        self.assertEqual(app.bar(None, None, 5), '.....')


if __name__ == '__main__':
    unittest.main()
