"""Schema linking for raw tables: table kind + column roles.

Three evidence sources, in priority order:
  1. the shop README data dictionary ("- **Header**: description") when present,
  2. header synonyms (folded, punctuation-free), interpreted *within* the table kind,
  3. value sniffing (ID patterns like ORD000123 / PN00012 / SP0001, plates, lot codes).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..textnorm import fold, fold_key
from .reader import RawTable

# ----------------------------------------------------------------------------- kinds
KINDS = ["sales", "orders", "payments", "products", "units", "stock", "purchases", "purchase_lines",
         "lots", "lot_sales", "deliveries", "debt_terms", "price_history", "expenses", "customers",
         "drugs", "returns", "tables", "stock_card", "staff", "suppliers", "shift_report", "aliases", "other"]


def _k(*names: str) -> List[str]:
    return [fold_key(n) for n in names]


# role -> header synonyms, per kind.  "*" = shared default for every kind.
SYN: Dict[str, Dict[str, List[str]]] = {
    "*": {
        "order_id": _k("Mã đơn", "Ma don", "Mã đơn hàng", "Mã hóa đơn", "Mã bill", "Bill", "Mã chứng từ", "Số CT",
                       "Số chứng từ", "Mã giao dịch", "So hoa don", "Số hóa đơn", "Order ID", "Đơn số", "HĐ", "Số HĐ",
                       "MaHoaDon", "Số đơn", "Số bill", "Mã GD", "Mã phiếu", "Số phiếu"),
        "line_no": _k("STT", "Số thứ tự", "No.", "Số dòng", "TT dòng", "Thứ tự", "Dòng", "SoThuTu"),
        "product_code": _k("Mã hàng", "Mã hàng hóa", "Mã SP", "Mã sản phẩm", "Mã SKU", "SKU", "Mã HH", "Mã H.hóa",
                           "ID SP", "Ma hang", "Mã hàng bán", "Mã mặt hàng", "Mã sản phẩm bán", "Mã vạch", "Barcode",
                           "MaHangHoa", "ma_sp", "Mã món", "Mã sp", "Mã barcode"),
        "product_name": _k("Tên hàng", "Tên hàng hóa", "Tên sản phẩm", "Tên SP", "Sản phẩm", "Hàng", "Hàng bán",
                           "Hàng hóa", "Mặt hàng", "Nội dung", "Product", "Item", "Ten hang", "TenHang", "Món",
                           "Chi tiết hàng", "Tên món"),
        "customer_id": _k("Mã KH", "Mã khách", "Mã khách hàng", "Customer ID", "Mã khách mua", "Mã người mua", "Mã thẻ",
                          "Số KH", "ma_kh", "MaKhachHang"),
        "customer_name": _k("Khách hàng", "Tên khách hàng", "Tên khách", "Khách", "Bên mua", "Đối tác", "Họ tên khách",
                            "Khách hàng (tên)", "TenKH", "Tên KH", "ten_kh", "KH"),
        "customer_group": _k("Nhóm khách hàng", "Nhóm khách", "Loại khách", "Hạng", "LoaiKhachHang", "Nhóm KH"),
        "phone": _k("Điện thoại", "Số điện thoại", "SĐT", "Số ĐT", "Phone", "Di động", "SoDienThoai"),
        "address": _k("Địa chỉ", "Addr", "DiaChiLienHe"),
        "note": _k("Ghi chú", "Ghi chu", "Note", "Lưu ý", "DienGiai", "Ghi chú KH"),
        "staff": _k("Người bán", "Nhân viên bán", "Mã NV bán", "MaNhanVien", "Mã NV", "Nhân viên"),
        "channel": _k("Kênh bán", "Bán qua", "Kênh đặt", "Loại bán", "Nguồn", "Nguồn đơn", "KenhBanHang", "Kênh"),
        "payment_method": _k("Phương thức thanh toán", "Phương thức TT", "Hình thức thanh toán", "Thanh toán bằng",
                             "Trả bằng", "PTTT", "HTTT", "Loại TT", "Phương thức", "PhuongThucTT", "PhuongThuc",
                             "Hình thức TT"),
        "status": _k("Trạng thái", "Trạng thái đơn", "Trạng thái đơn hàng", "Trạng thái xử lý", "Kết quả", "Status",
                     "Ghi chú TT", "TT đơn", "Tiến độ", "Tình trạng", "Tình trạng đơn", "Xử lý", "TinhTrang",
                     "Trạng thái giao"),
        "date": _k("Ngày bán", "Ngày", "Ngày tháng", "Ngày giao dịch", "Ngày lập", "Ngày mua", "Ngày xuất",
                   "Ngày đặt hàng", "NgayBan", "Thời gian", "Date", "Ngày HĐ", "NgayLap", "ngay_ban", "Ngày sửa"),
        "datetime": _k("Thời gian tạo", "Giờ bán", "Giờ tạo đơn", "ThoiGianLap", "Thời điểm đặt", "Ngày giờ"),
        "time": _k("Giờ"),
        "table_no": _k("Bàn", "Phòng/Bàn"),
        "area": _k("Khu vực"),
        "guests": _k("Số khách"),
        "empty": _k("Trạng thái HĐĐT", "Giới tính", "SoHoaDonDienTu", "TinhTrangHDDT"),
    },
    "sales": {
        "qty": _k("Số lượng", "SL", "SL bán", "SLg", "Sluong", "SoLuong", "Số lượng bán", "Số lượng hàng",
                  "Số lượng sản phẩm", "so_luong", "S.L", "Slg"),
        "unit": _k("Đơn vị tính", "ĐVT", "DVT", "Đơn vị", "ĐV tính", "Unit", "DonVi", "Đơn vị (tính)", "Đ.vị", "ĐV",
                   "DonViTinh", "dvt"),
        "base_qty": _k("Số lượng quy đổi", "SL quy đổi", "Quy ra ĐV lẻ", "Số lượng lẻ", "SL lẻ", "SL cơ bản",
                       "SLQuyDoi", "SL (ĐV cơ bản)", "SL quy ra lẻ"),
        "unit_price": _k("Đơn giá", "Giá", "Giá bán", "Giá bán lẻ", "Giá lẻ", "Giá sản phẩm", "Giá đơn vị", "Price",
                         "don_gia", "ĐG", "Đơn giá (đ)", "Giá 1 đv", "Giá niêm yết", "GiaDonVi", "Đơn giá bán"),
        "amount": _k("Thành tiền", "Thanh tien", "Thành tiền (đ)", "Tổng tiền", "Tổng", "Trị giá", "Tiền", "Tiền bán",
                     "Amount", "T.Tiền", "Cộng", "thanh_tien", "TriGia", "TT"),
        "line_discount": _k("Giảm giá", "Giảm (đ)", "Tiền giảm", "Chiết khấu", "Khuyến mãi", "ChietKhau"),
        "cost_unit": _k("Giá vốn", "Giá vốn lúc bán", "GiaVonLucBan"),
        "goods_total": _k("Tổng tiền hàng", "Tiền hàng", "Tien hang", "Giá trị hàng", "Tiền hàng trước giảm"),
        "order_discount": _k("Giảm giá HĐ", "Bớt", "Tiền giảm HĐ", "CK"),
        "shipping_fee": _k("Phí ship", "Phi ship", "Phí vận chuyển", "Phí giao"),
        "order_total": _k("Tổng sau giảm giá", "Tổng tiền đơn", "Tổng bill", "Tổng đơn", "Phải thu",
                          "Tổng thanh toán", "Tổng phiếu"),
        "paid": _k("Khách đã trả", "Khách trả", "Đã trả", "Da tra", "Đã thu"),
        "remaining": _k("Còn nợ", "Con no", "Dư nợ", "Chưa trả", "Công nợ"),
        "trade_in": _k("Thu máy cũ"),
        "imei": _k("IMEI"),
        "product_group": _k("Nhóm hàng", "Phân loại", "Nhóm hàng hóa", "Loại hàng"),
        "work": _k("Làm gì", "Công việc"),
        "vehicle": _k("Xe"),
        "mechanic": _k("Thợ"),
    },
    "orders": {
        "goods_total": _k("Tổng tiền hàng", "Tiền hàng", "Tien hang", "Giá trị hàng", "Tiền hàng trước giảm",
                          "TongTienHang"),
        "order_discount": _k("Giảm giá", "CK", "Chiết khấu", "Tiền giảm", "ChietKhau", "Bớt", "Giảm giá HĐ"),
        "shipping_fee": _k("Phí ship", "Phi ship", "Phí vận chuyển", "PhiVanChuyen", "Phí giao"),
        "vat": _k("Thuế", "Thuế GTGT", "VAT", "ThueGTGT"),
        "order_total": _k("Tổng sau giảm giá", "Tổng tiền đơn", "Tổng bill", "Tổng đơn", "Phải thu", "TongTien",
                          "Tong tien", "Tổng thanh toán", "TongThanhToan", "Thành tiền", "Tổng tiền", "Tổng phiếu"),
        "paid": _k("Khách đã trả", "Khách trả", "Đã trả", "Da tra", "Đã thu", "Thanh toán", "KhachDaTra"),
        "remaining": _k("Còn nợ", "Con no", "Còn lại", "Dư nợ", "Chưa trả", "Nợ", "Công nợ", "ConPhaiThu"),
        "shift": _k("MaCaLam", "Ca"),
        "promo": _k("MaKhuyenMai", "Mã khuyến mãi"),
        "branch": _k("MaChiNhanh"),
    },
    "payments": {
        "payment_id": _k("Mã phiếu thu", "Mã TT", "Phiếu thu", "Mã thanh toán", "MaPhieuThu", "Mã phiếu", "Số CT",
                         "Mã GD"),
        "pay_datetime": _k("Thời gian thu", "Ngày thu", "Ngay thu", "Thời điểm", "Ngày TT", "NgayGioThu", "Thời gian",
                           "Ngày"),
        "pay_amount": _k("Tiền", "Giá trị", "Số tiền", "So tien", "Số tiền thu", "Tiền thu", "SoTienThu"),
    },
    "products": {
        "product_name": _k("Tên hàng", "Tên hàng hóa", "Tên sản phẩm", "Tên SP", "Sản phẩm", "Hàng bán", "Hàng hóa",
                           "Mặt hàng", "Món", "TenGoi", "Tên", "Tên món", "Hàng", "Item"),
        "product_group": _k("Nhóm hàng", "Nhóm hàng(3 Cấp)", "Danh mục", "Loại hàng", "Loại sản phẩm", "Nhom hang",
                            "Nhóm", "Phân loại", "DanhMuc", "Nhóm hàng hóa"),
        "base_unit": _k("ĐVT", "Đơn vị", "Đơn vị tính", "Đơn vị cơ bản", "DonViCoSo", "DVT", "ĐV gốc"),
        "price": _k("Giá bán", "Giá", "Giá hiện tại", "Giá niêm yết", "Giá lẻ", "GiaBanLe"),
        "cost": _k("Giá vốn", "Giá vốn hiện tại", "GiaVonHang"),
        "active": _k("Đang kinh doanh", "ConKinhDoanh"),
        "stock_qty": _k("Tồn cuối", "Tồn kho", "Tồn cuối kỳ"),
        "brand": _k("Hãng", "Thương hiệu"),
        "size": _k("Số / Size", "Size", "Số/Size"),
        "aliases": _k("Tên gọi khác"),
        "vat_rate": _k("TyLeThue"),
        "conversion": _k("Quy đổi"),
        "product_type": _k("Loại hàng hóa"),
        "created": _k("Thời gian tạo", "TaoLuc", "Ngày tạo"),
    },
    "units": {
        "product_name": _k("Tên sản phẩm", "Hàng", "Tên món", "Tên hàng", "Sản phẩm"),
        "big_unit": _k("Đơn vị quy đổi", "ĐV lớn", "ĐV nhập", "Loại ĐV", "ĐVT", "Đơn vị", "DonViTinh", "Quy cách"),
        "factor": _k("Tỉ lệ quy đổi", "Hệ số", "Hệ số QĐ", "Quy đổi", "Bằng", "HeSoQuyDoi", "Tỷ lệ quy đổi"),
        "base_unit": _k("Đơn vị cơ bản", "Đơn vị gốc", "ĐV gốc", "ĐV nhỏ"),
        "unit_price": _k("Giá bán", "Giá bán theo đơn vị quy đổi", "GiaDonVi"),
    },
    "stock": {
        "product_name": _k("Tên hàng", "Hàng hóa", "Product", "Tên sản phẩm", "ten_hang", "Sản phẩm"),
        "unit": _k("ĐVT", "Đơn vị tính", "Đơn vị", "ĐV gốc"),
        "stock_qty": _k("Tồn cuối", "Tồn cuối kỳ", "Tồn thực", "Ton kho", "Tồn kho"),
    },
    "purchases": {
        "receipt_id": _k("Mã nhập hàng", "Mã phiếu nhập", "Mã đơn nhập", "Số phiếu", "MaPhieuNhap"),
        "date": _k("Ngày nhập", "Thời gian", "NgayNhapKho", "Ngày"),
        "supplier_id": _k("Mã NCC", "Mã nhà cung cấp", "MaNhaCungCap"),
        "supplier_name": _k("Nhà cung cấp", "Tên NCC"),
        "total": _k("Cần trả NCC", "Tổng tiền phiếu", "TongGiaTriNhap", "Tổng phiếu"),
        "status": _k("Trạng thái", "Trạng thái phiếu", "TinhTrang"),
        "note": _k("Ghi chú", "Nội dung", "Note", "Lưu ý", "DienGiai", "Ghi chu"),
    },
    "purchase_lines": {
        "receipt_id": _k("Mã nhập hàng", "Mã phiếu nhập", "Mã đơn nhập", "Số phiếu", "MaPhieuNhap"),
        "date": _k("Ngày nhập", "Thời gian", "Ngày"),
        "supplier_id": _k("Mã NCC", "Mã nhà cung cấp"),
        "supplier_name": _k("Nhà cung cấp"),
        "total": _k("Cần trả NCC", "Tổng tiền phiếu", "Tổng phiếu"),
        "status": _k("Trạng thái", "Trạng thái phiếu"),
        "note": _k("Ghi chú", "Nội dung", "Note", "Lưu ý", "Ghi chu", "Mô tả", "Diễn giải"),
        "product_name": _k("Tên mặt hàng", "Product", "Hàng hóa", "Tên hàng", "Hàng", "Tên sản phẩm"),
        "unit": _k("ĐVT", "ĐVT nhập", "Đơn vị", "DonViTinh"),
        "qty": _k("Số lượng", "SL nhập", "SL", "Số lượng nhập"),
        "unit_cost": _k("Đơn giá", "Giá nhập", "DonGiaNhap"),
        "amount": _k("Thành tiền"),
        "lot": _k("Lô", "Mã lô", "MaLo"),
        "expiry": _k("Hạn sử dụng", "HSD", "HanSuDung"),
    },
    "lots": {
        "receipt_id": _k("Mã nhập hàng", "Mã phiếu nhập", "Mã đơn nhập", "MaPhieuNhap"),
        "lot": _k("Lô", "Mã lô", "MaLo"),
        "expiry": _k("Hạn sử dụng", "HSD", "HanSuDung"),
        "qty_base": _k("Số lượng (ĐVT cơ bản)", "SL nhập (ĐVT cơ bản)", "Số lượng nhập", "SLNhapQuyDoi", "Số lượng"),
        "unit_cost": _k("DonGiaNhap", "Đơn giá nhập", "Giá nhập"),
    },
    "lot_sales": {
        "lot": _k("Lô", "Mã lô", "MaLo"),
        "qty_base": _k("Số lượng (ĐVT cơ bản)", "SL quy ra lẻ", "Số lượng", "SLQuyDoi"),
        "qty_returned": _k("Số lượng trả (ĐVT cơ bản)", "Số lượng trả", "SL trả (ĐVT cơ bản)", "SLTraQuyDoi"),
    },
    "deliveries": {
        "trip_no": _k("Lần giao", "Lượt giao", "Đợt giao"),
        "date": _k("Ngày giao hàng", "Ngày chở", "Thời gian giao", "Ngày giao"),
        "unit": _k("ĐVT", "ĐV gốc", "Đơn vị tính"),
        "qty_base": _k("Số lượng giao", "SL thực giao"),
        "plate": _k("Biển số xe"),
        "driver": _k("Người giao hàng", "Người chở", "Lái xe", "Tài xế"),
    },
    "debt_terms": {
        "days": _k("Số ngày được nợ", "Hạn nợ (ngày)", "Kỳ hạn"),
        "due_date": _k("Hạn thanh toán", "Han tra", "Ngày phải trả"),
    },
    "price_history": {
        "date": _k("Ngày áp dụng", "NgayApDung"),
        "new_price": _k("Giá mới", "GiaBanLe"),
        "old_price": _k("Giá cũ"),
        "product_name": _k("ten_hang", "TenHang", "Item", "Sản phẩm", "Diễn giải", "Tên hàng", "Hàng"),
    },
    "expenses": {
        "date": _k("Ngày", "Ngày mua", "Tháng", "Ngày chi"),
        "amount": _k("Tiền chi", "So tien", "Số tiền", "Tiền"),
        "category": _k("Nhóm", "Loại", "Khoản", "Khoản chi", "Hạng mục", "Mục"),
        "note": _k("Ghi chú", "Lưu ý", "Nội dung", "Diễn giải", "Mô tả"),
    },
    "customers": {
        "customer_name": _k("Tên khách hàng", "Tên", "Khách hàng (tên)", "TenGoi", "Tên KH", "ten_kh", "KH", "Khách hàng mua",
                            "Khách lẻ/sỉ", "Họ tên"),
        "customer_group": _k("Nhóm khách hàng", "Nhóm", "Loại khách", "Hạng", "LoaiKhachHang", "Kiểu khách"),
        "registered": _k("Ngày đăng ký"),
        "points": _k("Điểm hiện có", "Điểm hiện tại"),
        "baby_name": _k("Tên bé"),
        "created": _k("Ngày tạo", "TaoLuc"),
    },
    "drugs": {
        "active_ingredient": _k("Hoạt chất", "ThanhPhanChinh"),
        "rx": _k("Thuốc kê đơn", "Kê đơn", "BanTheoDon"),
    },
    "returns": {
        "return_id": _k("MaPhieuTra", "Mã phiếu trả"),
        "ret_datetime": _k("NgayGioTra", "Ngày trả", "Thời gian trả"),
        "reason": _k("NguyenNhan", "Lý do"),
        "refund": _k("TienHoanTra", "Tiền hoàn"),
        "refund_method": _k("CachHoanTien"),
    },
    "stock_card": {
        "move_datetime": _k("NgayGioPhatSinh"),
        "move_type": _k("LoaiGiaoDich"),
        "qty_base": _k("SLQuyDoi"),
        "doc_type": _k("KieuChungTu"),
        "doc_id": _k("MaChungTu"),
        "balance": _k("TonCuoi"),
    },
    "tables": {},
    "aliases": {"alias": _k("Ghi tắt", "Viết tắt", "Tên viết tắt"), "product_name": _k("Tên hàng", "Tên đầy đủ", "Tên"),
                "product_group": _k("Loại", "Loại hàng", "Nhóm")},
    "shift_report": {
        "date": _k("Ngày"), "shift": _k("Ca"), "cash_in": _k("Tiền mặt thu"), "cash_refund": _k("Tiền mặt hoàn khách"),
        "cash_net": _k("Tiền mặt ròng của ca"),
    },
    "staff": {"staff_name": _k("TenGoi", "Họ tên"), "role": _k("ChucVu"), "start": _k("NgayNhanViec", "Ngày vào làm"),
              "commission": _k("Hoa hồng (%)")},
    "suppliers": {"supplier_id": _k("MaNhaCungCap", "Mã NCC"), "supplier_name": _k("TenGoi", "Tên NCC", "Nhà cung cấp")},
}

# README description (folded) -> role ; checked with `in`/regex on the folded description
DESC_RULES: List[Tuple[str, str]] = [
    (r"^so thu tu dong", "line_no"),
    (r"^ma don$|^so hoa don|^so thu tu don(\s*\(| trong|$)", "order_id"),
    (r"^ten hang \(file chi co", "product_name"),
    (r"^ten hang|^hang ban", "product_name"),
    (r"^ma hang", "product_code"),
    (r"^so luong ban lay tu lo", "qty_base"),
    (r"^so luong khach tra lai", "qty_returned"),
    (r"^so luong nhap \(don vi co ban\)", "qty_base"),
    (r"^so luong nhap", "qty"),
    (r"^so luong da doi ra don vi co ban", "base_qty"),
    (r"^so luong", "qty"),
    (r"^don vi co ban", "base_unit"),
    (r"^don vi tinh", "unit"),
    (r"^don vi lon", "big_unit"),
    (r"^don vi nhap", "unit"),
    (r"^he so", "factor"),
    (r"^quy doi don vi lon", "conversion"),
    (r"^ngay ban", "date"),
    (r"^ngay sua", "date"),
    (r"^thoi diem dat don", "datetime"),
    (r"^gio dat don", "time"),
    (r"^don gia", "unit_price"),
    (r"^gia nhap", "unit_cost"),
    (r"^gia ban hien tai", "price"),
    (r"^gia von", "cost"),
    (r"^gia ban ap dung", "new_price"),
    (r"^gia truoc lan doi", "old_price"),
    (r"^ngay bat dau ap dung", "date"),
    (r"^trang thai don", "status"),
    (r"^trang thai phieu nhap", "status"),
    (r"^thanh tien sau giam|^thanh tien cua dong", "amount"),
    (r"^tien nhap cua ca dong", "amount"),
    (r"^ten khach", "customer_name"),
    (r"^ma khach", "customer_id"),
    (r"^nhom khach", "customer_group"),
    (r"^nhom hang", "product_group"),
    (r"^ten$", "name"),
    (r"^ma phieu nhap", "receipt_id"),
    (r"^ngay nhap", "date"),
    (r"^so lo", "lot"),
    (r"^han su dung", "expiry"),
    (r"^so dien thoai", "phone"),
    (r"^dia chi", "address"),
    (r"^ghi chu", "note"),
    (r"^cot trong", "empty"),
    (r"^giam gia cua ca don", "order_discount"),
    (r"^tien bot cho ca don", "order_discount"),
    (r"^tien giam cua ca dong", "line_discount"),
    (r"^giam gia$", "discount"),
    (r"^da thanh toan|^khach da tra|^so tien khach tra", "paid"),
    (r"^con no|^khach con no|^so khach con no", "remaining"),
    (r"^tong tien hang", "goods_total"),
    (r"^tien hang cua don", "goods_total"),
    (r"^tong tien don|^khach can tra cho ca don|^so khach phai tra cho ca don", "order_total"),
    (r"^tong tien phieu nhap|^tong tien cua ca phieu", "total"),
    (r"^tong tien ca phieu", "amount"),
    (r"^kenh ban", "channel"),
    (r"^hinh thuc thanh toan", "payment_method"),
    (r"^ma nhan vien ban", "staff"),
    (r"^thoi diem thu", "pay_datetime"),
    (r"^so tien$", "pay_amount"),
    (r"^ma phieu thu", "payment_id"),
    (r"^ma nha cung cap", "supplier_id"),
    (r"^nha cung cap", "supplier_name"),
    (r"^ton kho", "stock_qty"),
    (r"^ban$", "table_no"),
    (r"^khu vuc", "area"),
    (r"^so khach cua don", "guests"),
    (r"^phi giao hang", "shipping_fee"),
    (r"^bien so xe", "plate"),
    (r"^dot giao", "trip_no"),
    (r"^ngay giao", "date"),
    (r"^tai xe", "driver"),
    (r"^so ngay duoc no", "days"),
    (r"^han thanh toan", "due_date"),
    (r"^hoat chat", "active_ingredient"),
    (r"^thuoc ke don", "rx"),
    (r"^con kinh doanh", "active"),
    (r"^ngay tao", "created"),
    (r"^imei", "imei"),
    (r"^tien thu may cu", "trade_in"),
    (r"^dong xe", "vehicle"),
    (r"^cac viec", "work"),
    (r"^tho lam phieu", "mechanic"),
    (r"^ho ten tho", "staff_name"),
    (r"^ngay vao lam", "start"),
    (r"^phan tram hoa hong", "commission"),
    (r"^ten goi khac", "aliases"),
    (r"^hang$", "brand"),
    (r"^so cua sua", "size"),
    (r"^ten be", "baby_name"),
    (r"^ngay dang ky", "registered"),
    (r"^diem con", "points"),
    (r"^ten chuong trinh", "promo_name"),
    (r"^hang ap dung", "promo_items"),
    (r"^tu ngay", "date_from"),
    (r"^den ngay", "date_to"),
]

# role translation by kind for README generic roles
_GENERIC_BY_KIND = {
    "name": {"products": "product_name", "customers": "customer_name", "*": "product_name"},
    "discount": {"orders": "order_discount", "sales": "line_discount", "*": "line_discount"},
    "cost": {"sales": "cost_unit", "*": "cost"},
    "unit_price": {"purchase_lines": "unit_cost", "units": "unit_price", "*": "unit_price"},
    "qty": {"*": "qty"},
    "unit": {"units": "big_unit", "stock": "unit", "*": "unit"},
    "base_unit": {"stock": "unit", "*": "base_unit"},
    "amount": {"*": "amount"},
    "stock_qty": {"*": "stock_qty"},
    "product_group": {"*": "product_group"},
}

# ----------------------------------------------------------------------------- value sniffing
_ID_PATTERNS = {
    "ORD": "order_id", "HD": "order_id", "PN": "receipt_id", "PAY": "payment_id", "PT": "payment_id",
    "SP": "product_code", "KH": "customer_id", "NCC": "supplier_id", "NV": "staff", "TH": "return_id",
    "MV": "move_id", "CA": "shift", "CN": "branch", "KM": "promo",
}


def sniff_column(values: pd.Series) -> Optional[str]:
    vals = [str(v).strip() for v in values.head(400) if v is not None and str(v).strip() not in ("", "nan", "None")]
    if len(vals) < 3:
        return None
    pref = Counter()
    for v in vals:
        m = re.fullmatch(r"([A-Z]{1,4})(\d{3,})", v)
        if m:
            pref[m.group(1)] += 1
    if pref:
        p, c = pref.most_common(1)[0]
        if c >= 0.8 * len(vals) and p in _ID_PATTERNS:
            return _ID_PATTERNS[p]
    if sum(bool(re.fullmatch(r"\d{2}[A-Z]{1,2}-\d{3}\.\d{2}", v)) for v in vals) >= 0.8 * len(vals):
        return "plate"
    if sum(bool(re.fullmatch(r"[A-Z]\d{7}", v)) for v in vals) >= 0.8 * len(vals):
        return "lot"
    return None


# ----------------------------------------------------------------------------- README parsing
def parse_readme_columns(readme: str) -> Dict[str, Dict[str, str]]:
    """{file_or_sheet_key: {header: description}} from '## Cột trong `X`[, sheet `Y`]' sections."""
    out: Dict[str, Dict[str, str]] = {}
    cur = None
    for line in readme.splitlines():
        m = re.match(r"^##\s+Cột trong\s+`([^`]+)`(?:\s*,\s*sheet\s+`([^`]+)`)?", line.strip())
        if m:
            cur = (m.group(1) + ("::" + m.group(2) if m.group(2) else ""))
            out[cur] = {}
            continue
        if line.startswith("## "):
            cur = None
            continue
        if cur is not None:
            m = re.match(r"^\s*-\s+\*\*(.+?)\*\*\s*:\s*(.+)$", line)
            if m:
                out[cur][m.group(1).strip()] = m.group(2).strip()
    return out


def desc_to_role(desc: str) -> Optional[str]:
    d = fold(desc)
    for rx, role in DESC_RULES:
        if re.search(rx, d):
            return role
    return None


# ----------------------------------------------------------------------------- kind classification
_NAME_HINTS = [
    ("lot_sales", ["chitietlo", "bantheolo", "donhanglo", "chitietdonlo", "hoadonchitietlo"]),
    ("stock_card", ["thekho"]),
    ("tables", ["phongban"]),
    ("returns", ["trahang"]),
    ("deliveries", ["giaohang", "phieugiao"]),
    ("debt_terms", ["congnotheohan", "hanno", "sohanno"]),
    ("price_history", ["banggia", "lichsugia"]),
    ("expenses", ["chiphi"]),
    ("drugs", ["danhsachthuoc", "thongtinthuoc", "banhangthuoc"]),
    ("lots", ["lohang", "danhsachlo"]),
    ("units", ["quydoi", "donvitinh", "donviquydoi", "donvisp"]),
    ("stock", ["tonkho"]),
    ("purchase_lines", ["chitietnhap", "nhaphangchitiet", "sonhaphang", "sotaynhaphang", "nhaphang"]),
    ("purchases", ["phieunhap", "danhsachnhaphang", "donnhaphang"]),
    ("payments", ["thanhtoan", "phieuthu", "soquy"]),
    ("customers", ["khachhang"]),
    ("suppliers", ["nhacungcap"]),
    ("staff", ["nhanvien", "danhsachtho"]),
    ("shift_report", ["baocaoca"]),
    ("products", ["sanpham", "hanghoa", "danhmuc", "danhmuchang"]),
    ("sales", ["chitiet", "sobanhang", "banhang", "chitietbanhang", "baocaobanhang", "donhangchitiet", "tonghop",
               "sosuaxe"]),
    ("orders", ["hoadon", "donhang", "danhsachhoadon"]),
]


def _header_roles(cols: List[str], kind: str) -> Dict[str, str]:
    table = dict(SYN["*"])
    table.update(SYN.get(kind, {}))
    inv: Dict[str, str] = {}
    for role, syns in table.items():
        for s in syns:
            inv.setdefault(s, role)
    # kind-specific synonyms take precedence
    for role, syns in SYN.get(kind, {}).items():
        for s in syns:
            inv[s] = role
    out = {}
    for c in cols:
        r = inv.get(fold_key(re.sub(r"\.\d+$", "", c)))
        if r:
            out[c] = r
    return out


def classify(t: RawTable, readme_cols: Optional[Dict[str, str]] = None) -> str:
    name_key = fold_key((t.sheet or "") + " " + t.file.rsplit(".", 1)[0])
    file_key = fold_key(t.file.rsplit(".", 1)[0])
    sheet_key = fold_key(t.sheet or "")
    cols = t.columns
    sniff = {c: sniff_column(t.df[c]) for c in cols}
    sniffed = set(v for v in sniff.values() if v)
    hk = set(fold_key(re.sub(r"\.\d+$", "", c)) for c in cols)
    roles_any = set(_header_roles(cols, "sales").values()) | set(_header_roles(cols, "orders").values())
    if readme_cols:
        for h, d in readme_cols.items():
            r = desc_to_role(d)
            if r:
                roles_any.add(r)

    def has(*keys):
        return any(fold_key(k) in hk for k in keys)

    # strong structural signals first
    if "lot" in sniffed and "order_id" in sniffed and not has("Tên hàng", "Đơn giá"):
        return "lot_sales"
    if has("NgayGioPhatSinh") and has("LoaiGiaoDich", "TonCuoi"):
        return "stock_card"
    if has("MaPhieuTra", "NgayGioTra"):
        return "returns"
    if has("Ghi tắt", "Viết tắt", "Tên viết tắt") and len(cols) <= 5:
        return "aliases"
    if has("Hoạt chất", "ThanhPhanChinh") and len(cols) <= 5:
        return "drugs"
    if has("Biển số xe") and has("Lần giao", "Lượt giao"):
        return "deliveries"
    if has("Số ngày được nợ", "Hạn nợ (ngày)", "Kỳ hạn") and len(cols) <= 5:
        return "debt_terms"
    if has("Giá mới") and has("Ngày áp dụng", "NgayApDung"):
        return "price_history"
    if has("NgayApDung") and has("GiaBanLe") and len(cols) <= 5:
        return "price_history"
    if has("Phòng/Bàn", "Bàn") and has("Số khách") and len(cols) <= 5:
        return "tables"
    if has("Tiền mặt thu", "Tiền mặt ròng của ca"):
        return "shift_report"
    if "lot" in sniffed and has("Hạn sử dụng", "HSD", "HanSuDung") and "receipt_id" in sniffed and \
            not has("Đơn giá", "Giá nhập", "DonGiaNhap", "SL", "Số lượng nhập", "ĐVT") :
        return "lots"
    if has("SLNhapQuyDoi"):
        return "lots"
    if "payment_id" in sniffed and "order_id" in sniffed:
        return "payments"
    if "receipt_id" in sniffed:
        has_prod = has("Mã hàng", "Mã SKU", "MaHangHoa", "Product", "Tên mặt hàng", "Hàng hóa", "Hàng", "Tên hàng",
                       "Mã hàng bán", "Tên sản phẩm") or "product_code" in sniffed
        if has_prod:
            if ("lot" in sniffed and has("Hạn sử dụng", "HSD")) and not has("Đơn giá", "Giá nhập", "DonGiaNhap", "SL", "ĐVT", "Số lượng nhập", "SL nhập"):
                return "lots"
            return "purchase_lines"
        return "purchases"
    if has("Tồn cuối", "Tồn cuối kỳ", "Tồn thực", "Ton kho") and not ("order_id" in sniffed):
        if has("Giá vốn", "Giá", "Nhóm", "Giá bán", "Đang kinh doanh"):
            return "products"
        return "stock"
    if has("Tỉ lệ quy đổi", "Hệ số", "Hệ số QĐ", "HeSoQuyDoi", "Bằng", "Tỷ lệ quy đổi") or \
            (has("Quy đổi") and has("ĐVT", "Mã hàng") and len(cols) <= 5):
        return "units"
    if has("Tiền chi") or ("chiphi" in name_key):
        return "expenses"
    if "order_id" in sniffed or ("order_id" in roles_any and not "receipt_id" in sniffed):
        line_signal = has("Tên hàng", "Tên hàng hóa", "Tên sản phẩm", "Mã hàng", "Sản phẩm", "Hàng", "Hàng bán",
                          "Mặt hàng", "Product", "Nội dung", "Chi tiết hàng", "Ten hang", "Tên SP", "MaHangHoa",
                          "Mã mặt hàng", "Mã hàng bán", "ID SP", "Ma hang", "Hàng hóa", "Mã sản phẩm bán", "Món",
                          "Tên món", "Mã SKU", "SKU") or "product_code" in sniffed or \
            any(r in roles_any for r in ("product_name", "product_code"))
        if line_signal:
            return "sales"
        if "staff" in sniffed and not has("Tổng tiền hàng", "Tổng sau giảm giá", "Khách đã trả"):
            pass
        return "orders"
    if "customer_id" in sniffed and not ("product_code" in sniffed):
        return "customers"
    if "supplier_id" in sniffed and has("TenGoi"):
        return "suppliers"
    if "staff" in sniffed and has("TenGoi", "ChucVu"):
        return "staff"
    # no ids: rely on names
    for kind, keys in _NAME_HINTS:
        for k in keys:
            if k in sheet_key or k in file_key:
                if kind in ("sales", "orders"):
                    if has("Tên hàng", "Hàng", "Mặt hàng", "Nội dung", "Ten hang", "Sản phẩm", "Hàng bán", "Làm gì",
                           "Tên sản phẩm") or any(r in roles_any for r in ("product_name",)):
                        return "sales"
                    continue
                return kind
    if has("Họ tên") and has("Hoa hồng (%)"):
        return "staff"
    if has("Tên KH", "Tên khách hàng") and has("SĐT", "Điện thoại", "Số điện thoại"):
        return "customers"
    if any(r in roles_any for r in ("product_name",)) and any(r in roles_any for r in ("qty", "amount", "date")):
        return "sales"
    if has("Làm gì", "Xe"):
        return "sales"
    return "other"


def assign_roles(t: RawTable, kind: str, readme_cols: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return {column: role}."""
    roles = _header_roles(t.columns, kind)
    if readme_cols:
        for c in t.columns:
            base = re.sub(r"\.\d+$", "", c)
            d = readme_cols.get(base)
            if d is None:
                # tolerate case / accent differences in README header names
                for h, dd in readme_cols.items():
                    if fold_key(h) == fold_key(base):
                        d = dd
                        break
            if d is None:
                continue
            r = desc_to_role(d)
            if r is None:
                continue
            if r in _GENERIC_BY_KIND:
                m = _GENERIC_BY_KIND[r]
                r = m.get(kind, m.get("*", r))
            if kind == "products" and r == "unit":
                r = "base_unit"
            if kind == "stock" and r == "product_code":
                r = "product_code"
            if kind in ("purchases", "purchase_lines") and r == "date":
                r = "date"
            if kind == "sales" and r == "total":
                r = "order_total"
            if kind == "payments" and r in ("amount", "pay_amount"):
                r = "pay_amount"
            roles[c] = r
    # value sniffing overrides for id-like columns
    for c in t.columns:
        s = sniff_column(t.df[c])
        if not s:
            continue
        if s == "staff" and kind in ("sales", "orders"):
            roles[c] = "staff"
        elif s in ("order_id", "receipt_id", "payment_id", "product_code", "customer_id", "supplier_id", "lot",
                   "plate", "return_id"):
            if s == "order_id" and kind == "payments" and roles.get(c) == "payment_id":
                roles[c] = "order_id"
            elif s == "payment_id" and kind == "payments":
                roles[c] = "payment_id"
            elif roles.get(c) in (None, "empty", "note", "line_no", "order_id", "payment_id", "receipt_id",
                                  "product_name", "customer_name", "staff", "product_code", "customer_id"):
                if not (s == "product_code" and roles.get(c) == "product_code"):
                    roles[c] = s
    # de-duplicate: keep the first column for unique roles, prefer columns with data
    seen: Dict[str, str] = {}
    sniffed = {c: sniff_column(t.df[c]) for c in roles}
    for c, r in list(roles.items()):
        if r == "line_discount" and r in seen and kind == "sales":
            # two discount columns in a line export: the order-level one is 'Tiền giảm' / 'CK' (repeated per order)
            prev = seen[r]
            hp = fold_key(re.sub(r"\.\d+$", "", prev))
            if hp in ("tiengiam", "ck", "chietkhau", "giamgiahd"):
                roles[prev] = "order_discount"
                seen[r] = c
            else:
                roles[c] = "order_discount"
            continue
        if r in seen:
            prev = seen[r]
            if sniffed.get(c) == r and sniffed.get(prev) != r:
                roles.pop(prev)
                seen[r] = c
                continue
            if sniffed.get(prev) == r and sniffed.get(c) != r:
                roles.pop(c)
                continue
            nn_prev = t.df[prev].map(lambda v: v is not None and str(v).strip() != "").mean()
            nn_cur = t.df[c].map(lambda v: v is not None and str(v).strip() != "").mean()
            if nn_cur > nn_prev + 0.2:
                roles.pop(prev)
                seen[r] = c
            else:
                roles.pop(c)
        else:
            seen[r] = c
    return roles
