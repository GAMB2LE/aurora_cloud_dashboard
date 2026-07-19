import unittest

import instrument_registry
import mobile_catalog


class InstrumentRegistryTests(unittest.TestCase):
    def test_assigned_pdu_contract_is_stable(self):
        self.assertEqual(
            [
                (
                    instrument.id,
                    instrument.pdu_title,
                    instrument.pdu_outlet,
                )
                for instrument in instrument_registry.PDU_INSTRUMENTS
            ],
            [
                ("uas", "UAS", 4),
                ("ceilometer", "CL61", 5),
                ("cloud-radar", "Cloud Radar", 6),
                ("hatpro", "HATPRO", 8),
            ],
        )

    def test_mobile_catalog_uses_shared_contract(self):
        self.assertIs(mobile_catalog.INSTRUMENTS, instrument_registry.INSTRUMENTS)
        self.assertIs(mobile_catalog.INSTRUMENT_BY_ID, instrument_registry.INSTRUMENT_BY_ID)

    def test_browser_options_preserve_existing_labels(self):
        self.assertEqual(
            instrument_registry.browser_options(),
            {
                "Ceilometer": "Ceilometer",
                "Cloud Radar": "Cloud Radar",
                "Meteorology": "vaisalamet",
                "Radiation": "asfs-logger",
                "Aurora Power Supply": "power",
                "WXcam": "wxcam",
                "Scanning Microwave Radiometer": "Scanning Microwave Radiometer",
            },
        )
        self.assertEqual(
            instrument_registry.browser_options(housekeeping=True)["Operations"],
            "ops-monitor",
        )


if __name__ == "__main__":
    unittest.main()
